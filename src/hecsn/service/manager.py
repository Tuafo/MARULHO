from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import Event, RLock, Thread
import time
from typing import Any, Iterator, Mapping, Sequence, cast
from uuid import uuid4

import torch

from hecsn.config.presets import get_autonomy_acquisition_preset
from hecsn.config.model_config import HECSNConfig
from hecsn.data.corpus_loader import StreamingCorpusLoader
from hecsn.data.dataset_adapters import (
    NMNISTAdapter,
    FSDDAdapter,
    PairedDigitDataset,
    iter_episode_steps,
)
from hecsn.data.event_camera_encoder import EventCameraEncoder
from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.data.rtf_encoder import RTFEncoder
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
DEFAULT_AUTONOMY_REMOTE_PROVIDERS: tuple[str, ...] = ("wikipedia", "arxiv", "openalex")
DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT = 4
DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT = 4
DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER = 2
DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT = 4
AUTO_REMOTE_QUERY_BUDGET_MAX = 4
AUTO_REMOTE_PROVIDER_PRIORITY_WEIGHT = 0.35
AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT = 8
AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT = 4
AUTO_FOCUS_SHORTLIST_MAX_SIZE = 3
AUTO_FOCUS_SHORTLIST_GAP_WEIGHT = 0.2
AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT = 0.8

TERMINUS_QUICK_START_PRESETS: dict[str, dict[str, Any]] = {
    "wikipedia": {
        "label": "Wikipedia (wikitext-103)",
        "description": "General knowledge from English Wikipedia. Good all-around starting point.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "wikipedia_news": {
        "label": "Wikipedia + News",
        "description": "Wikipedia paired with AG News for broader topic coverage.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "diverse": {
        "label": "Diverse (Wiki + News + Reviews)",
        "description": "Three domains for maximum coverage: Wikipedia, AG News, and IMDB reviews.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "diverse_fast": {
        "label": "Diverse — Fast (Wiki + News + Reviews)",
        "description": "Same three domains but with larger batches and shorter sleep for faster throughput. Use when you want quicker training at the cost of slightly chunkier UI updates.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 1024,
        "sleep_interval_seconds": 0.02,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 2048, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 8, "plasticity_spike_backend": "adex"},
    },
    "multimodal": {
        "label": "Multimodal (Wiki + N-MNIST + FSDD)",
        "description": "Text from Wikipedia interleaved with N-MNIST visual and FSDD audio digit episodes. Trains all three modalities.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "multimodal": {
            "enabled": True,
            "nmnist_dir": "N-MNIST",
            "fsdd_dir": "free-spoken-digit-dataset-master",
            "episode_interval_tokens": 256,
            "n_steps": 10,
        },
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex", "enable_cross_modal": True, "cross_modal_dim_visual": 64, "cross_modal_dim_audio": 64},
    },
    "multimodal_fast": {
        "label": "Multimodal — Fast",
        "description": "Faster multimodal training with larger batches. Good for scale testing with all three modalities active.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "multimodal": {
            "enabled": True,
            "nmnist_dir": "N-MNIST",
            "fsdd_dir": "free-spoken-digit-dataset-master",
            "episode_interval_tokens": 128,
            "n_steps": 10,
        },
        "tick_tokens": 1024,
        "sleep_interval_seconds": 0.02,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 2048, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 8, "plasticity_spike_backend": "adex", "enable_cross_modal": True, "cross_modal_dim_visual": 64, "cross_modal_dim_audio": 64},
    },
}

_IRREGULAR_TOPIC_SINGULARS = {
    "octopi": "octopus",
    "octopuses": "octopus",
}


def _canonical_provider_term(value: Any) -> str:
    normalized = " ".join(str(value).split()).strip().lower()
    if not normalized:
        return ""

    def _canonical_token(token: str) -> str:
        if token in _IRREGULAR_TOPIC_SINGULARS:
            return _IRREGULAR_TOPIC_SINGULARS[token]
        if token in {"species", "series"} or len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
            return token[:-2]
        if token.endswith("oes") and len(token) > 4:
            return token[:-2]
        if token.endswith(("us", "is")):
            return token
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    return " ".join(_canonical_token(part) for part in normalized.split() if part).strip()


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


class HECSNServiceManager:
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
        self._brain_source_index = 0
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
        self._brain_thread: Thread | None = None
        self._brain_stop_event: Event | None = None
        self._brain_running = False
        self._multimodal_dataset: PairedDigitDataset | None = None
        self._multimodal_step_iter: Iterator | None = None
        self._multimodal_visual_encoder: EventCameraEncoder | None = None
        self._multimodal_audio_encoder: CochleagramEncoder | None = None
        self._multimodal_tokens_since_episode = 0
        self._multimodal_episodes_completed = 0
        self._multimodal_visual_accepted = 0
        self._multimodal_audio_accepted = 0
        self._rebuild_brain_sources_locked()
        self._dirty_state = False
        self._state_revision = 0
        self._load_persisted_traces_locked()

        # --- Cortex / ThoughtLoop (lazy, graceful degradation) ---
        self._thought_loop: Any = None  # type: ThoughtLoop | None
        self._cortex_available = False
        try:
            from hecsn.cortex.thought_loop import ThoughtLoop
            from hecsn.cortex.core import CorticalCore

            cortex = CorticalCore()  # will fail if Ollama unreachable
            self._thought_loop = ThoughtLoop(cortex=cortex)
            self._cortex_available = True
            _cortex_logger.info("Cortex module initialised (Ollama cortex)")
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

    def terminus_status(self) -> dict[str, Any]:
        # Non-blocking: return cached data when brain loop holds the lock
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            cached = getattr(self, "_cached_terminus_status", None)
            if cached is not None:
                return cached
            self._lock.acquire()
        try:
            mm_enabled = self._multimodal_dataset is not None
            mm_info = {
                "enabled": mm_enabled,
                "episodes_completed": int(self._multimodal_episodes_completed),
                "tokens_since_episode": int(self._multimodal_tokens_since_episode),
            }
            result = {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
                "multimodal": mm_info,
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
        multimodal: dict[str, Any] | None = None,
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
                    "multimodal": multimodal,
                }
            )
            self._brain_last_error = None
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

    def quick_start_terminus(self, *, preset: str = "wikipedia") -> dict[str, Any]:
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
            multimodal=config.get("multimodal"),
        )
        result = self.start_terminus()
        result["already_running"] = False
        result["preset_applied"] = preset
        return result

    @staticmethod
    def quick_start_presets() -> list[dict[str, Any]]:
        """Return available quick-start presets for the UI."""
        return [
            {"id": key, "label": val["label"], "description": val["description"], "source_count": len(val["source_bank"])}
            for key, val in TERMINUS_QUICK_START_PRESETS.items()
        ]

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

    def architecture_summary(self) -> dict[str, Any]:
        """Return a runtime-driven description of the active model architecture."""
        with self._lock:
            model = self._trainer.model
            config = self._trainer.config
            layers: list[dict[str, Any]] = []
            layers.append({
                "id": "input_encoding",
                "name": "Input Encoding (RTF)",
                "enabled": True,
                "type": "encoder",
                "params": {
                    "input_dim": int(config.input_dim),
                    "representation": config.input_representation,
                    "learned_chunking": bool(config.enable_learned_chunking),
                },
            })
            layers.append({
                "id": "competitive_routing",
                "name": "Competitive Column Routing",
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
                "id": "context_prediction",
                "name": f"Context Prediction ({config.context_mode})",
                "enabled": model.context_layer is not None,
                "type": "context",
                "params": {
                    "context_mode": config.context_mode,
                },
            })
            layers.append({
                "id": "binding",
                "name": "Temporal Binding (STP)",
                "enabled": model.binding_layer is not None,
                "type": "binding",
                "params": {
                    "n_bindings": int(config.binding_n_bindings),
                    "fan_in": int(config.binding_fan_in),
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
                "name": "Cross-Modal Grounding",
                "enabled": model.cross_modal is not None,
                "type": "grounding",
                "params": {
                    "dim_visual": int(config.cross_modal_dim_visual),
                    "dim_audio": int(config.cross_modal_dim_audio),
                    "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()) if model.cross_modal else 0.0,
                    "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()) if model.cross_modal else 0.0,
                },
            })
            layers.append({
                "id": "memory_consolidation",
                "name": "Dual Memory Store",
                "enabled": True,
                "type": "memory",
                "params": {
                    "memory_capacity": int(config.memory_capacity),
                    "stc_tag_duration_strong": float(config.stc_tag_duration_strong),
                },
            })
            return {
                "model_name": "HECSNModel",
                "version": "v4",
                "layers": layers,
                "config": {
                    "context_mode": config.context_mode,
                    "plasticity_rule": config.plasticity_rule,
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
            "multimodal": self._normalize_multimodal_config(config.get("multimodal")),
        }
        return normalized

    @staticmethod
    def _normalize_multimodal_config(config: Any) -> dict[str, Any] | None:
        if config is None or not isinstance(config, dict):
            return None
        if not config.get("enabled"):
            return None
        return {
            "enabled": True,
            "nmnist_dir": str(config.get("nmnist_dir", "N-MNIST")),
            "fsdd_dir": str(config.get("fsdd_dir", "free-spoken-digit-dataset-master")),
            "episode_interval_tokens": max(32, int(config.get("episode_interval_tokens", 256))),
            "n_steps": max(1, int(config.get("n_steps", 10))),
        }

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=str(spec.get("source_type", "auto")),
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        return labeled_pattern_stream(
            loader.char_stream(),
            self._encoder,
            self._trainer.config.window_size,
            learn_chunking=True,
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

    def _init_multimodal_locked(self) -> None:
        """Initialize multimodal dataset + encoders if the config enables them."""
        mm = self._brain_config.get("multimodal")
        if not mm or not mm.get("enabled"):
            self._multimodal_dataset = None
            self._multimodal_step_iter = None
            self._multimodal_visual_encoder = None
            self._multimodal_audio_encoder = None
            self._multimodal_tokens_since_episode = 0
            self._multimodal_episodes_completed = 0
            self._multimodal_visual_accepted = 0
            self._multimodal_audio_accepted = 0
            return

        nmnist_dir = Path(mm.get("nmnist_dir", "N-MNIST"))
        fsdd_dir = Path(mm.get("fsdd_dir", "free-spoken-digit-dataset-master"))
        n_steps = int(mm.get("n_steps", 10))

        # Resolve relative paths from CWD
        if not nmnist_dir.is_absolute():
            nmnist_dir = Path.cwd() / nmnist_dir
        if not fsdd_dir.is_absolute():
            fsdd_dir = Path.cwd() / fsdd_dir

        # NMNISTAdapter expects root with Train/ subdir, FSDDAdapter expects root with recordings/ subdir
        train_dir = nmnist_dir / "Train"
        rec_dir = fsdd_dir / "recordings"
        if not train_dir.exists() or not rec_dir.exists():
            self._multimodal_dataset = None
            self._multimodal_step_iter = None
            return

        model = self._trainer.model
        # EventCameraEncoder(34,34,pool=4) → 8×8 = 64 output dim
        # CochleagramEncoder(n_bands=64) → 64 output dim
        # Cross-modal layer must match these real encoder dims

        try:
            nmnist = NMNISTAdapter(nmnist_dir)
            fsdd = FSDDAdapter(fsdd_dir)
            dataset = PairedDigitDataset(nmnist, fsdd, n_steps=n_steps)
            self._multimodal_dataset = dataset
            self._multimodal_visual_encoder = EventCameraEncoder(
                height=34, width=34, pool=4,
            )
            self._multimodal_audio_encoder = CochleagramEncoder(
                sample_rate=8000, n_bands=64,
            )
            self._multimodal_step_iter = iter_episode_steps(
                dataset.iter_episodes(),
                visual_encoder=self._multimodal_visual_encoder,
                audio_encoder=self._multimodal_audio_encoder,
            )
            self._multimodal_tokens_since_episode = 0
            self._multimodal_episodes_completed = 0
            self._multimodal_visual_accepted = 0
            self._multimodal_audio_accepted = 0
        except Exception as exc:
            _cortex_logger.warning("Multimodal init failed: %s", exc)
            self._multimodal_dataset = None
            self._multimodal_step_iter = None

    def _run_multimodal_episode_locked(self) -> dict[str, Any] | None:
        """Run one multimodal episode if due. Returns summary or None."""
        mm = self._brain_config.get("multimodal")
        if not mm or not mm.get("enabled") or self._multimodal_step_iter is None:
            return None

        interval = int(mm.get("episode_interval_tokens", 256))
        if self._multimodal_tokens_since_episode < interval:
            return None

        self._multimodal_tokens_since_episode = 0
        steps_trained = 0
        last_metrics = None

        try:
            n_steps = int(mm.get("n_steps", 10))
            for _ in range(n_steps):
                try:
                    step = next(self._multimodal_step_iter)
                except StopIteration:
                    # Re-create iterator (loop over dataset)
                    if self._multimodal_dataset is not None:
                        self._multimodal_step_iter = iter_episode_steps(
                            self._multimodal_dataset.iter_episodes(),
                            visual_encoder=self._multimodal_visual_encoder,
                            audio_encoder=self._multimodal_audio_encoder,
                        )
                        try:
                            step = next(self._multimodal_step_iter)
                        except StopIteration:
                            break
                    else:
                        break

                # Encode text label using blended_feature_vector (same path
                # as iter_char_patterns used by the regular brain loop).
                codes = [ord(c) if 0 <= ord(c) < 128 else 0 for c in step.text]
                pattern = self._encoder.blended_feature_vector(codes)
                last_metrics = self._trainer.train_step(
                    pattern,
                    raw_window=step.text,
                    visual_spikes=step.visual_spikes,
                    audio_spikes=step.audio_spikes,
                )
                steps_trained += 1
                if last_metrics:
                    if last_metrics.get("cross_modal_visual_accepted"):
                        self._multimodal_visual_accepted += 1
                    if last_metrics.get("cross_modal_audio_accepted"):
                        self._multimodal_audio_accepted += 1

            self._multimodal_episodes_completed += 1
            self._mark_mutated()

            return {
                "type": "multimodal_episode",
                "steps_trained": steps_trained,
                "episodes_completed": self._multimodal_episodes_completed,
                "last_metrics": last_metrics,
            }
        except Exception as exc:
            _cortex_logger.warning("Multimodal episode failed: %s", exc, exc_info=True)
            return None

    def _rebuild_brain_sources_locked(self) -> None:
        self._close_brain_sources_locked()
        self._brain_source_runtimes = [
            _BrainSourceRuntime(spec=deepcopy(spec), stream=self._build_brain_source_stream_locked(spec))
            for spec in self._brain_config.get("source_bank", [])
        ]
        self._brain_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._init_multimodal_locked()

    def _request_brain_stop(self, *, reason: str | None = None) -> Thread | None:
        with self._lock:
            return self._request_brain_stop_locked(reason=reason)

    def _join_brain_thread(self, thread: Thread | None, *, timeout: float = 5.0) -> None:
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
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=str(spec.get("source_type", "auto")),
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
            self._multimodal_tokens_since_episode += total_trained
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

        multimodal_summary = self._run_multimodal_episode_locked()
        autonomy_summary = self._run_brain_autonomy_locked()
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or multimodal_summary is not None

        # Inject SNN neuromodulator state into cortex drives
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

        multimodal_summary = self._run_multimodal_episode_locked()
        autonomy_summary = self._run_brain_autonomy_locked()
        did_work = autonomy_summary is not None or multimodal_summary is not None
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": {"did_work": False, "reason": "sources_exhausted"},
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": 0,
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = 0
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
        multimodal_summary = self._run_multimodal_episode_locked()
        autonomy_summary = self._run_brain_autonomy_locked()
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or multimodal_summary is not None
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
            self._multimodal_tokens_since_episode += token_count
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
            "multimodal": {
                "enabled": bool(self._multimodal_dataset is not None),
                "episodes_completed": int(self._multimodal_episodes_completed),
                "tokens_since_episode": int(self._multimodal_tokens_since_episode),
                "episode_interval": int(
                    (self._brain_config.get("multimodal") or {}).get("episode_interval_tokens", 256)
                ),
                "cross_modal_visual_accepted": int(self._multimodal_visual_accepted),
                "cross_modal_audio_accepted": int(self._multimodal_audio_accepted),
            },
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

    def _autonomy_focus_plan_locked(self) -> dict[str, Any] | None:
        recent_query_focus = self._recent_query_focus_plan_locked()
        abstraction_query = ""
        if recent_query_focus is not None:
            abstraction_query = " ".join(
                [
                    *[
                        str(value)
                        for value in list(recent_query_focus.get("query_terms") or [])[:4]
                        if str(value).strip()
                    ],
                    *[
                        str(value)
                        for value in list(recent_query_focus.get("unsupported_terms") or [])[:2]
                        if str(value).strip()
                    ],
                ]
            ).strip()
        concept_focus = (
            self._concept_store.focus_plan(
                query_text=abstraction_query,
                min_observations=1,
            )
            if abstraction_query
            else self._concept_store.focus_plan()
        )
        geometric_focus = self._geometric_curiosity.focus_plan(
            query_text=abstraction_query or None,
        )
        plans = [
            plan
            for plan in (recent_query_focus, concept_focus, geometric_focus)
            if isinstance(plan, Mapping)
        ]
        if not plans:
            return None
        merged: dict[str, Any] = deepcopy(dict(plans[0]))
        for plan in plans[1:]:
            merged = self._merge_focus_plans_locked(merged, plan)
        return merged

    def _merge_focus_plans_locked(
        self,
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
    ) -> dict[str, Any]:
        def _dedupe(values: Sequence[str], limit: int) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for raw_value in values:
                value = " ".join(str(raw_value).split()).strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                ordered.append(value)
                if len(ordered) >= max(1, int(limit)):
                    break
            return ordered

        gap_weights: Counter[str] = Counter()
        unsupported_weights: Counter[str] = Counter()
        query_terms: list[str] = []
        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concept_scores: dict[str, dict[str, Any]] = {}
        geometric_gaps: list[dict[str, Any]] = []

        for plan, plan_weight in ((primary, 1.0), (secondary, 0.75)):
            query_terms.extend(str(term) for term in list(plan.get("query_terms") or []) if str(term).strip())
            retrieval_queries.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
            follow_up_questions.extend(str(item) for item in list(plan.get("follow_up_questions") or []) if str(item).strip())
            for raw_gap in list(plan.get("geometric_gaps") or []):
                if isinstance(raw_gap, Mapping):
                    geometric_gaps.append(deepcopy(dict(raw_gap)))

            for raw_term in list(plan.get("unsupported_terms") or []):
                term = str(raw_term).strip().lower()
                if not term:
                    continue
                unsupported_weights[term] += float(plan_weight)
                gap_weights[term] += float(plan_weight)

            for raw_gap in list(plan.get("gap_terms") or []):
                if not isinstance(raw_gap, Mapping):
                    continue
                term = str(raw_gap.get("term", "")).strip().lower()
                if not term:
                    continue
                gap_weights[term] += float(plan_weight) * max(0.0, float(raw_gap.get("weight", 0.0)))

            for raw_concept in list(plan.get("weak_concepts") or []):
                if not isinstance(raw_concept, Mapping):
                    continue
                label = " ".join(str(raw_concept.get("label", "")).split()).strip()
                top_terms = [
                    " ".join(str(value).split()).strip().lower()
                    for value in list(raw_concept.get("top_terms") or [])
                    if " ".join(str(value).split()).strip()
                ]
                if not label and not top_terms:
                    continue
                key = label.lower() if label else "|".join(top_terms[:3])
                if not key:
                    continue
                aggregate = weak_concept_scores.setdefault(
                    key,
                    {
                        "label": label,
                        "top_terms": [],
                        "weight_sum": 0.0,
                        "weakness_sum": 0.0,
                        "uncertainty_sum": 0.0,
                        "drift_sum": 0.0,
                        "match_count": 0,
                    },
                )
                aggregate["label"] = str(aggregate["label"] or label)
                aggregate["top_terms"] = list(dict.fromkeys([*list(aggregate["top_terms"]), *top_terms]))[:4]
                aggregate["weight_sum"] = float(aggregate["weight_sum"]) + float(plan_weight)
                aggregate["weakness_sum"] = float(aggregate["weakness_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("weakness", 0.0)),
                )
                aggregate["uncertainty_sum"] = float(aggregate["uncertainty_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("uncertainty", 0.0)),
                )
                aggregate["drift_sum"] = float(aggregate["drift_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("drift", 0.0)),
                )
                aggregate["match_count"] = max(
                    int(aggregate["match_count"]),
                    max(0, int(raw_concept.get("match_count", 0))),
                )

        unsupported_terms = [
            term
            for term, _weight in sorted(
                unsupported_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:8]
        ]
        if not retrieval_queries and unsupported_terms:
            retrieval_queries.append(" ".join(unsupported_terms[:3]))

        weak_concepts = [
            {
                "label": str(values["label"]),
                "weakness": float(values["weakness_sum"]) / max(1e-8, float(values["weight_sum"])),
                "uncertainty": float(values["uncertainty_sum"]) / max(1e-8, float(values["weight_sum"])),
                "drift": float(values["drift_sum"]) / max(1e-8, float(values["weight_sum"])),
                "top_terms": list(values["top_terms"])[:4],
                "match_count": int(values["match_count"]),
            }
            for _key, values in sorted(
                weak_concept_scores.items(),
                key=lambda item: (
                    -(float(item[1]["weakness_sum"]) / max(1e-8, float(item[1]["weight_sum"]))),
                    -(float(item[1]["uncertainty_sum"]) / max(1e-8, float(item[1]["weight_sum"]))),
                    str(item[1]["label"] or "|".join(list(item[1]["top_terms"]))),
                ),
            )[:4]
        ]
        structural_growth = None
        secondary_growth = secondary.get("structural_growth")
        primary_growth = primary.get("structural_growth")
        if isinstance(secondary_growth, Mapping):
            structural_growth = deepcopy(dict(secondary_growth))
        elif isinstance(primary_growth, Mapping):
            structural_growth = deepcopy(dict(primary_growth))

        merged = {
            "planner_mode": "merged_runtime_abstraction_focus",
            "query_terms": _dedupe(query_terms, 8),
            "unsupported_terms": unsupported_terms,
            "gap_terms": [
                {"term": term, "weight": float(weight)}
                for term, weight in sorted(
                    gap_weights.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:8]
            ],
            "retrieval_queries": _dedupe(retrieval_queries, 4),
            "follow_up_questions": _dedupe(follow_up_questions, 4),
            "weak_concepts": weak_concepts,
        }
        if structural_growth is not None:
            merged["structural_growth"] = structural_growth
        if geometric_gaps:
            merged["geometric_gaps"] = geometric_gaps[:4]
        return merged

    def _recent_query_focus_plan_locked(self) -> dict[str, Any] | None:
        if not self._brain_recent_query_gaps:
            return None
        gap_weights: Counter[str] = Counter()
        unsupported_weights: Counter[str] = Counter()
        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concept_scores: dict[str, dict[str, Any]] = {}
        query_terms: list[str] = []
        seen_queries: set[str] = set()
        seen_questions: set[str] = set()
        seen_terms: set[str] = set()
        for index, item in enumerate(list(self._brain_recent_query_gaps)):
            recency_weight = 1.0 / float(index + 1)
            for raw_term in salient_query_terms(str(item.get("query_text", ""))):
                term = str(raw_term).strip().lower()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                query_terms.append(term)
            for raw_gap in list(item.get("gap_terms") or []):
                if not isinstance(raw_gap, dict):
                    continue
                term = str(raw_gap.get("term", "")).strip().lower()
                if not term:
                    continue
                gap_weights[term] += recency_weight * max(0.0, float(raw_gap.get("weight", 0.0)))
            for raw_term in list(item.get("unsupported_terms") or []):
                term = str(raw_term).strip().lower()
                if not term:
                    continue
                unsupported_weights[term] += recency_weight
                gap_weights[term] += 2.0 * recency_weight
                if term not in seen_terms:
                    seen_terms.add(term)
                    query_terms.append(term)
            for raw_query in list(item.get("retrieval_queries") or []):
                retrieval_query = " ".join(str(raw_query).split()).strip()
                if not retrieval_query:
                    continue
                lowered = retrieval_query.lower()
                if lowered in seen_queries:
                    continue
                seen_queries.add(lowered)
                retrieval_queries.append(retrieval_query)
            for raw_question in list(item.get("follow_up_questions") or []):
                question = " ".join(str(raw_question).split()).strip()
                if not question:
                    continue
                lowered = question.lower()
                if lowered in seen_questions:
                    continue
                seen_questions.add(lowered)
                follow_up_questions.append(question)
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
                key = label.lower() if label else "|".join(top_terms[:3])
                if not key:
                    continue
                aggregate = weak_concept_scores.setdefault(
                    key,
                    {
                        "label": label,
                        "top_terms": [],
                        "weight_sum": 0.0,
                        "weakness_sum": 0.0,
                        "uncertainty_sum": 0.0,
                        "drift_sum": 0.0,
                        "match_count": 0,
                    },
                )
                aggregate["label"] = str(aggregate["label"] or label)
                aggregate["top_terms"] = list(
                    dict.fromkeys([*list(aggregate["top_terms"]), *top_terms])
                )[:4]
                aggregate["weight_sum"] = float(aggregate["weight_sum"]) + recency_weight
                aggregate["weakness_sum"] = float(aggregate["weakness_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("weakness", 0.0)),
                )
                aggregate["uncertainty_sum"] = float(aggregate["uncertainty_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("uncertainty", 0.0)),
                )
                aggregate["drift_sum"] = float(aggregate["drift_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("drift", 0.0)),
                )
                aggregate["match_count"] = max(
                    int(aggregate["match_count"]),
                    max(0, int(raw_concept.get("match_count", 0))),
                )
        if not gap_weights and not unsupported_weights and not retrieval_queries and not follow_up_questions and not weak_concept_scores:
            return None
        unsupported_terms = [
            term
            for term, _weight in sorted(
                unsupported_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:8]
        ]
        if not retrieval_queries and unsupported_terms:
            retrieval_queries.append(" ".join(unsupported_terms[:3]))
        weak_concepts = [
            {
                "label": str(values["label"]),
                "weakness": (
                    float(values["weakness_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "uncertainty": (
                    float(values["uncertainty_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "drift": (
                    float(values["drift_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "top_terms": list(values["top_terms"])[:4],
                "match_count": int(values["match_count"]),
            }
            for _key, values in sorted(
                weak_concept_scores.items(),
                key=lambda item: (
                    -(
                        float(item[1]["weakness_sum"])
                        / max(1e-8, float(item[1]["weight_sum"]))
                    ),
                    -(
                        float(item[1]["uncertainty_sum"])
                        / max(1e-8, float(item[1]["weight_sum"]))
                    ),
                    str(item[1]["label"] or "|".join(list(item[1]["top_terms"]))),
                ),
            )[:4]
        ]
        return {
            "planner_mode": "recent_query_gap_focus",
            "query_terms": query_terms[:8],
            "unsupported_terms": unsupported_terms,
            "gap_terms": [
                {"term": term, "weight": float(weight)}
                for term, weight in sorted(
                    gap_weights.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:8]
            ],
            "retrieval_queries": retrieval_queries[:4],
            "follow_up_questions": follow_up_questions[:4],
            "weak_concepts": weak_concepts,
        }

    def _autonomy_candidate_specs_locked(
        self,
        *,
        candidate_bank: list[dict[str, Any]],
        focus_plan: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        specs = deepcopy(candidate_bank)
        if focus_plan is None:
            return specs
        retrieval_target_count = min(2, len(list(focus_plan.get("retrieval_queries") or [])))
        follow_up_target_count = 1 if list(focus_plan.get("follow_up_questions") or []) else 0
        curiosity_ready_weak_concepts = self._curiosity_ready_weak_concept_count_locked(focus_plan)
        focus_text = " ".join(
            [
                *[str(item) for item in list(focus_plan.get("query_terms") or [])[:3]],
                *[str(item) for item in list(focus_plan.get("retrieval_queries") or [])[:4]],
                *[str(item) for item in list(focus_plan.get("unsupported_terms") or [])[:3]],
            ]
        ).strip()
        if not focus_text:
            return specs
        for spec in specs:
            if str(spec.get("catalog_mode", "")).strip():
                existing_focus = " ".join(str(spec.get("catalog_focus_text", "")).split()).strip()
                if existing_focus and existing_focus.lower() != "none":
                    spec["catalog_focus_text"] = f"{existing_focus} {focus_text}".strip()
                else:
                    spec["catalog_focus_text"] = focus_text
                if str(spec.get("catalog_mode", "")).strip().lower() == "live_remote_search":
                    current_queries_per_provider = max(
                        1,
                        int(spec.get("catalog_queries_per_provider", DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER)),
                    )
                    desired_queries_per_provider = max(
                        current_queries_per_provider,
                        min(
                            AUTO_REMOTE_QUERY_BUDGET_MAX,
                            retrieval_target_count
                            + min(2, curiosity_ready_weak_concepts)
                            + follow_up_target_count,
                        ),
                    )
                    spec["catalog_queries_per_provider"] = int(desired_queries_per_provider)
                    self._apply_provider_curriculum_locked(spec, focus_plan=focus_plan)
                continue
            metadata = dict(spec.get("metadata") or {})
            existing_query_text = " ".join(str(metadata.get("query_text", "")).split()).strip()
            if existing_query_text and existing_query_text.lower() != "none":
                metadata["query_text"] = f"{existing_query_text} {focus_text}".strip()
            else:
                metadata["query_text"] = focus_text
            metadata["semantic_relevance"] = float(metadata.get("semantic_relevance", 0.0))
            spec["metadata"] = metadata
        return specs

    def _curiosity_ready_weak_concept_count_locked(self, focus_plan: Mapping[str, Any] | None) -> int:
        if focus_plan is None:
            return 0
        ready_count = 0
        for raw_concept in list(focus_plan.get("weak_concepts") or []):
            if not isinstance(raw_concept, Mapping):
                continue
            label = " ".join(str(raw_concept.get("label", "")).split()).strip()
            top_terms = [
                " ".join(str(value).split()).strip()
                for value in list(raw_concept.get("top_terms") or [])
                if " ".join(str(value).split()).strip()
            ]
            if not label and not top_terms:
                continue
            weakness = max(0.0, min(1.0, float(raw_concept.get("weakness", 0.0))))
            uncertainty = max(0.0, min(1.0, float(raw_concept.get("uncertainty", 0.0))))
            intermediate_uncertainty = max(0.0, 1.0 - min(1.0, abs(uncertainty - 0.5) / 0.5))
            curiosity_score = 0.65 * weakness + 0.35 * intermediate_uncertainty
            if curiosity_score >= 0.45:
                ready_count += 1
        return ready_count

    def _provider_curriculum_focus_terms_locked(self, focus_plan: Mapping[str, Any] | None) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        def _extend(values: Sequence[str]) -> None:
            for raw_value in values:
                for term in salient_query_terms(str(raw_value)):
                    normalized = _canonical_provider_term(term)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    ordered.append(normalized)
                    if len(ordered) >= AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT:
                        return

        if focus_plan is None:
            return []

        explicit_focus_signals = [
            str(item)
            for item in list(focus_plan.get("query_terms") or [])
            if str(item).strip()
        ]
        explicit_focus_signals.extend(
            str(item)
            for item in list(focus_plan.get("unsupported_terms") or [])
            if str(item).strip()
        )
        explicit_focus_signals.extend(
            str(item.get("term", ""))
            for item in list(focus_plan.get("gap_terms") or [])
            if isinstance(item, Mapping) and str(item.get("term", "")).strip()
        )
        _extend(explicit_focus_signals)
        if ordered:
            return ordered[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]

        fallback_focus_signals = [
            str(item)
            for item in list(focus_plan.get("focus_terms") or [])
            if str(item).strip()
        ]
        fallback_focus_signals.extend(
            str(item)
            for item in list(focus_plan.get("retrieval_queries") or [])
            if str(item).strip()
        )
        _extend(fallback_focus_signals)
        for raw_concept in list(focus_plan.get("weak_concepts") or []):
            if not isinstance(raw_concept, Mapping):
                continue
            _extend([str(item) for item in list(raw_concept.get("top_terms") or []) if str(item).strip()])
            if len(ordered) >= AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT:
                break
        return ordered[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]

    def _provider_topic_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        commits = max(0, int(family_entry.get("commits", 0)))
        successes = max(0, int(family_entry.get("successes", 0)))
        success_rate = 0.0 if commits <= 0 else float(successes) / float(commits)
        priority = float(
            0.32 * max(0.0, min(1.0, float(family_entry.get("answerability_gain_ema", 0.0))))
            + 0.22 * max(0.0, min(1.0, float(family_entry.get("uncertainty_reduction_ema", 0.0))))
            + 0.18 * max(0.0, min(1.0, float(family_entry.get("weak_concept_stabilization_ema", 0.0))))
            + 0.13 * max(0.0, min(1.0, float(family_entry.get("semantic_relevance_ema", 0.0))))
            + 0.10 * success_rate
            + 0.05 * min(1.0, float(commits) / 3.0)
        )
        return max(0.0, min(1.0, priority))

    def _provider_topic_family_match_score_locked(self, family_term: str, focus_terms: Sequence[str]) -> float:
        normalized_family = _canonical_provider_term(family_term)
        if not normalized_family:
            return 0.0
        family_tokens = {term.lower() for term in salient_query_terms(normalized_family)}
        if not family_tokens:
            family_tokens = {part for part in normalized_family.split() if part}
        if not family_tokens:
            return 0.0
        best = 0.0
        for raw_focus in focus_terms:
            normalized_focus = _canonical_provider_term(raw_focus)
            if not normalized_focus:
                continue
            if normalized_focus == normalized_family:
                return 1.0
            focus_tokens = {term.lower() for term in salient_query_terms(normalized_focus)}
            if not focus_tokens:
                focus_tokens = {part for part in normalized_focus.split() if part}
            if not focus_tokens:
                continue
            overlap = float(len(family_tokens & focus_tokens)) / float(max(len(family_tokens), len(focus_tokens)))
            if normalized_focus in normalized_family or normalized_family in normalized_focus:
                overlap = max(overlap, 0.75)
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    def _provider_topic_family_details_locked(
        self,
        entry: Mapping[str, Any],
        focus_terms: Sequence[str],
    ) -> tuple[float, list[str], int, dict[str, float], float]:
        raw_topic_families = entry.get("topic_families")
        if not focus_terms or not isinstance(raw_topic_families, Mapping):
            return 0.0, [], 0, {}, 0.0
        focus_index = {term: index for index, term in enumerate(focus_terms)}
        ranked_matches: list[tuple[float, str, int]] = []
        for raw_family, raw_family_entry in raw_topic_families.items():
            family = " ".join(str(raw_family).split()).strip().lower()
            if not family or not isinstance(raw_family_entry, Mapping):
                continue
            match_score = self._provider_topic_family_match_score_locked(family, focus_terms)
            if match_score <= 0.0:
                continue
            family_priority = self._provider_topic_family_priority_locked(raw_family_entry)
            if family_priority <= 0.0:
                continue
            commits = max(0, int(raw_family_entry.get("commits", 0)))
            ranked_matches.append((float(match_score * family_priority), family, commits))
        if not ranked_matches:
            return 0.0, [], 0, {}, 0.0
        ranked_matches.sort(
            key=lambda item: (
                -float(item[0]),
                int(focus_index.get(item[1], AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT)),
                item[1],
            )
        )
        top_matches = ranked_matches[: min(3, len(ranked_matches))]
        strength = max(
            0.0,
            min(
                1.0,
                sum(float(score) for score, _family, _commits in top_matches) / float(len(top_matches)),
            ),
        )
        best_score, _best_family, best_commits = top_matches[0]
        query_bonus = 0
        if best_commits >= 2 and best_score >= 0.28:
            query_bonus = 1
        if best_commits >= 4 and best_score >= 0.45:
            query_bonus = 2
        return (
            strength,
            [family for _score, family, _commits in top_matches],
            query_bonus,
            {family: float(score) for score, family, _commits in top_matches},
            float(best_score),
        )

    def _provider_query_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        commits = max(0, int(family_entry.get("commits", 0)))
        successes = max(0, int(family_entry.get("successes", 0)))
        success_rate = 0.0 if commits <= 0 else float(successes) / float(commits)
        priority = float(
            0.30 * max(0.0, min(1.0, float(family_entry.get("answerability_gain_ema", 0.0))))
            + 0.25 * max(0.0, min(1.0, float(family_entry.get("uncertainty_reduction_ema", 0.0))))
            + 0.20 * max(0.0, min(1.0, float(family_entry.get("weak_concept_stabilization_ema", 0.0))))
            + 0.15 * max(0.0, min(1.0, float(family_entry.get("semantic_relevance_ema", 0.0))))
            + 0.10 * success_rate
        )
        return max(0.0, min(1.0, priority))

    def _provider_query_family_match_score_locked(
        self,
        query_family: str,
        focus_terms: Sequence[str],
        focus_queries: Sequence[str],
    ) -> float:
        normalized_family = " ".join(str(query_family).split()).strip().lower()
        if not normalized_family:
            return 0.0
        family_tokens = {term.lower() for term in salient_query_terms(normalized_family)}
        if not family_tokens:
            family_tokens = {part for part in normalized_family.split() if part}
        if not family_tokens:
            return 0.0
        best = 0.0
        for raw_query in focus_queries:
            normalized_query = " ".join(str(raw_query).split()).strip().lower()
            if not normalized_query:
                continue
            if normalized_query == normalized_family:
                return 1.0
            query_tokens = {term.lower() for term in salient_query_terms(normalized_query)}
            if not query_tokens:
                query_tokens = {part for part in normalized_query.split() if part}
            if not query_tokens:
                continue
            overlap = float(len(family_tokens & query_tokens)) / float(max(len(family_tokens), len(query_tokens)))
            if normalized_query in normalized_family or normalized_family in normalized_query:
                overlap = max(overlap, 0.80)
            best = max(best, overlap)
        for raw_focus in focus_terms:
            normalized_focus = " ".join(str(raw_focus).split()).strip().lower()
            if not normalized_focus:
                continue
            focus_tokens = {term.lower() for term in salient_query_terms(normalized_focus)}
            if not focus_tokens:
                focus_tokens = {part for part in normalized_focus.split() if part}
            if not focus_tokens:
                continue
            overlap = float(len(family_tokens & focus_tokens)) / float(max(len(family_tokens), len(focus_tokens)))
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    def _provider_query_family_details_locked(
        self,
        entry: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
        focus_terms: Sequence[str],
    ) -> tuple[float, list[str], int, dict[str, float], float]:
        raw_query_families = entry.get("query_families")
        if not focus_terms or not isinstance(raw_query_families, Mapping):
            return 0.0, [], 0, {}, 0.0
        if not isinstance(focus_plan, Mapping) or not list(focus_plan.get("geometric_gaps") or []):
            return 0.0, [], 0, {}, 0.0
        focus_queries = [
            " ".join(str(item).split()).strip().lower()
            for item in list(focus_plan.get("retrieval_queries") or [])
            if " ".join(str(item).split()).strip()
        ]
        if not focus_queries:
            focus_queries = [
                " ".join(str(item).split()).strip().lower()
                for item in list(focus_plan.get("follow_up_questions") or [])
                if " ".join(str(item).split()).strip()
            ]
        ranked_matches: list[tuple[float, str, int]] = []
        for raw_family, raw_family_entry in raw_query_families.items():
            family = " ".join(str(raw_family).split()).strip().lower()
            if not family or not isinstance(raw_family_entry, Mapping):
                continue
            match_score = self._provider_query_family_match_score_locked(family, focus_terms, focus_queries)
            if match_score <= 0.0:
                continue
            family_priority = self._provider_query_family_priority_locked(raw_family_entry)
            if family_priority <= 0.0:
                continue
            commits = max(0, int(raw_family_entry.get("commits", 0)))
            ranked_matches.append((float(match_score * family_priority), family, commits))
        if not ranked_matches:
            return 0.0, [], 0, {}, 0.0
        ranked_matches.sort(key=lambda item: (-float(item[0]), item[1]))
        top_matches = ranked_matches[: min(3, len(ranked_matches))]
        strength = max(
            0.0,
            min(
                1.0,
                sum(float(score) for score, _family, _commits in top_matches) / float(len(top_matches)),
            ),
        )
        best_score, _best_family, best_commits = top_matches[0]
        query_bonus = 0
        if best_commits >= 1 and best_score >= 0.18:
            query_bonus = 1
        if best_commits >= 2 and best_score >= 0.35:
            query_bonus = 2
        return (
            strength,
            [family for _score, family, _commits in top_matches],
            query_bonus,
            {family: float(score) for score, family, _commits in top_matches},
            float(best_score),
        )

    def _provider_curriculum_priority_locked(
        self,
        provider: str,
        focus_plan: Mapping[str, Any] | None,
        *,
        autonomy: Mapping[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        normalized_provider = " ".join(str(provider).split()).strip().lower()
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        entry = curriculum.get(normalized_provider, {})
        attempts = max(0, int(entry.get("attempts", 0)))
        commits = max(0, int(entry.get("commits", 0)))
        successes = max(0, int(entry.get("successes", 0)))
        success_rate = 0.0 if attempts <= 0 else float(successes) / float(attempts)
        commit_rate = 0.0 if attempts <= 0 else float(commits) / float(attempts)
        diagnostic_gain = max(0.0, min(1.0, float(entry.get("diagnostic_gain_ema", 0.0))))
        semantic_relevance = max(0.0, min(1.0, float(entry.get("semantic_relevance_ema", 0.0))))
        answerability_gain = max(0.0, min(1.0, float(entry.get("answerability_gain_ema", 0.0))))
        uncertainty_reduction = max(0.0, min(1.0, float(entry.get("uncertainty_reduction_ema", 0.0))))
        weak_concept_stabilization = max(
            0.0,
            min(1.0, float(entry.get("weak_concept_stabilization_ema", 0.0))),
        )
        focus_terms = self._provider_curriculum_focus_terms_locked(focus_plan)
        topic_terms = {
            str(term).strip().lower(): float(weight)
            for term, weight in dict(entry.get("topic_terms") or {}).items()
            if str(term).strip() and float(weight) > 0.0
        }
        (
            topic_family_strength,
            matched_topic_families,
            topic_family_query_bonus,
            topic_family_scores,
            topic_family_focus_score,
        ) = self._provider_topic_family_details_locked(entry, focus_terms)
        (
            query_family_strength,
            matched_query_families,
            query_family_query_bonus,
            query_family_scores,
            query_family_focus_score,
        ) = self._provider_query_family_details_locked(entry, focus_plan, focus_terms)
        topic_overlap = 0.0
        if focus_terms and topic_terms:
            denominator = sum(float(weight) for weight in topic_terms.values())
            if denominator > 0.0:
                topic_overlap = max(
                    0.0,
                    min(
                        1.0,
                        sum(float(topic_terms.get(term, 0.0)) for term in focus_terms) / float(denominator),
                    ),
                )
        exploration_bonus = 0.0 if attempts > 0 else 0.15
        exploration_bonus += 0.10 / math.sqrt(float(attempts) + 1.0)
        priority = float(
            0.20 * success_rate
            + 0.13 * commit_rate
            + 0.15 * diagnostic_gain
            + 0.09 * semantic_relevance
            + 0.10 * answerability_gain
            + 0.07 * uncertainty_reduction
            + 0.08 * weak_concept_stabilization
            + 0.08 * topic_overlap
            + 0.10 * topic_family_strength
            + 0.20 * topic_family_focus_score
            + 0.06 * query_family_strength
            + 0.14 * query_family_focus_score
            + exploration_bonus
        )
        return priority, {
            "attempts": attempts,
            "commits": commits,
            "successes": successes,
            "success_rate": float(success_rate),
            "commit_rate": float(commit_rate),
            "diagnostic_gain_ema": float(entry.get("diagnostic_gain_ema", 0.0)),
            "semantic_relevance_ema": float(entry.get("semantic_relevance_ema", 0.0)),
            "answerability_gain_ema": float(entry.get("answerability_gain_ema", 0.0)),
            "uncertainty_reduction_ema": float(entry.get("uncertainty_reduction_ema", 0.0)),
            "weak_concept_stabilization_ema": float(entry.get("weak_concept_stabilization_ema", 0.0)),
            "topic_overlap": float(topic_overlap),
            "topic_family_strength": float(topic_family_strength),
            "topic_family_focus_score": float(topic_family_focus_score),
            "topic_family_query_bonus": int(topic_family_query_bonus),
            "matched_topic_families": list(matched_topic_families),
            "topic_family_scores": dict(topic_family_scores),
            "query_family_strength": float(query_family_strength),
            "query_family_focus_score": float(query_family_focus_score),
            "query_family_query_bonus": int(query_family_query_bonus),
            "matched_query_families": list(matched_query_families),
            "query_family_scores": dict(query_family_scores),
            "last_query_text": str(entry.get("last_query_text", "")),
            "last_selected_at": str(entry.get("last_selected_at", "")),
            "topic_terms": dict(entry.get("topic_terms") or {}),
            "topic_families": dict(entry.get("topic_families") or {}),
            "query_families": dict(entry.get("query_families") or {}),
        }

    def _provider_curriculum_snapshot_locked(
        self,
        autonomy: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        if not curriculum:
            return None
        ranked: list[dict[str, Any]] = []
        for provider in curriculum:
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            ranked.append(
                {
                    "provider": provider,
                    "priority": float(priority),
                    **details,
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["priority"]),
                -int(item["successes"]),
                str(item["provider"]),
            )
        )
        return {
            "focus_terms": self._provider_curriculum_focus_terms_locked(focus_plan),
            "ranked_providers": ranked[: max(1, len(ranked))],
        }

    def _apply_provider_curriculum_locked(
        self,
        spec: dict[str, Any],
        *,
        focus_plan: Mapping[str, Any] | None,
    ) -> None:
        autonomy = cast(dict[str, Any], self._brain_config.get("autonomy") or {})
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        if not curriculum:
            return
        providers = [
            str(provider).strip()
            for provider in list(spec.get("catalog_providers") or [])
            if str(provider).strip()
        ]
        if not providers:
            return
        ranked: list[tuple[int, str, float]] = []
        priority_map: dict[str, float] = {}
        provider_topic_terms: dict[str, list[str]] = {}
        provider_query_families: dict[str, list[str]] = {}
        topic_family_query_bonus = 0
        query_family_query_bonus = 0
        for index, provider in enumerate(providers):
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            ranked.append((index, provider, float(priority)))
            priority_map[str(provider)] = float(priority)
            curriculum_entry = curriculum.get(str(provider).strip().lower()) or {}
            matched_topic_families = [
                str(term).strip()
                for term in list(details.get("matched_topic_families") or [])
                if str(term).strip()
            ]
            topic_terms = [
                str(term).strip()
                for term in dict(curriculum_entry.get("topic_terms") or {}).keys()
                if str(term).strip()
            ]
            if matched_topic_families:
                ordered_terms = list(dict.fromkeys([*matched_topic_families, *topic_terms]))
            elif dict(curriculum_entry.get("topic_families") or {}):
                ordered_terms = []
            else:
                ordered_terms = list(topic_terms)
            if ordered_terms:
                provider_topic_terms[str(provider)] = ordered_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            topic_family_query_bonus = max(topic_family_query_bonus, int(details.get("topic_family_query_bonus", 0)))
            matched_query_families = [
                str(query).strip()
                for query in list(details.get("matched_query_families") or [])
                if str(query).strip()
            ]
            if matched_query_families:
                provider_query_families[str(provider)] = matched_query_families[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
            query_family_query_bonus = max(query_family_query_bonus, int(details.get("query_family_query_bonus", 0)))
        ranked.sort(key=lambda item: (-float(item[2]), int(item[0])))
        spec["catalog_providers"] = [provider for _index, provider, _priority in ranked]
        spec["catalog_provider_priority_map"] = dict(priority_map)
        if provider_topic_terms:
            spec["catalog_provider_topic_terms"] = dict(provider_topic_terms)
        if provider_query_families:
            spec["catalog_provider_query_families"] = dict(provider_query_families)
        spec["catalog_topic_family_budget_bonus"] = int(topic_family_query_bonus)
        spec["catalog_query_family_budget_bonus"] = int(query_family_query_bonus)
        total_query_bonus = int(topic_family_query_bonus) + int(query_family_query_bonus)
        if total_query_bonus > 0:
            spec["catalog_queries_per_provider"] = int(
                min(
                    AUTO_REMOTE_QUERY_BUDGET_MAX,
                    max(1, int(spec.get("catalog_queries_per_provider", 1))) + total_query_bonus,
                )
            )
        spec["catalog_provider_priority_weight"] = float(
            max(
                float(spec.get("catalog_provider_priority_weight", 0.0)),
                AUTO_REMOTE_PROVIDER_PRIORITY_WEIGHT,
            )
        )

    def _update_provider_curriculum_locked(
        self,
        *,
        autonomy: dict[str, Any],
        result: Mapping[str, Any],
        candidate_specs: Sequence[dict[str, Any]],
        focus_plan: Mapping[str, Any] | None,
    ) -> None:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))

        def _ensure(provider: str) -> dict[str, Any]:
            normalized_provider = " ".join(str(provider).split()).strip().lower()
            if not normalized_provider:
                return {}
            entry = curriculum.setdefault(
                normalized_provider,
                {
                    "attempts": 0,
                    "commits": 0,
                    "successes": 0,
                    "gap_gain_ema": 0.0,
                    "diagnostic_gain_ema": 0.0,
                    "semantic_relevance_ema": 0.0,
                    "answerability_gain_ema": 0.0,
                    "uncertainty_reduction_ema": 0.0,
                    "weak_concept_stabilization_ema": 0.0,
                    "last_query_text": "",
                    "last_selected_at": "",
                    "topic_terms": {},
                    "topic_families": {},
                    "query_families": {},
                },
            )
            entry["topic_terms"] = dict(entry.get("topic_terms") or {})
            entry["topic_families"] = dict(entry.get("topic_families") or {})
            entry["query_families"] = dict(entry.get("query_families") or {})
            return entry

        attempted_providers: list[str] = []
        for spec in candidate_specs:
            if str(spec.get("catalog_mode", "")).strip().lower() != "live_remote_search":
                continue
            attempted_providers.extend(
                str(provider).strip().lower()
                for provider in list(spec.get("catalog_providers") or [])
                if str(provider).strip()
            )
        for provider in dict.fromkeys(attempted_providers):
            entry = _ensure(provider)
            if entry:
                entry["attempts"] = int(entry.get("attempts", 0)) + 1

        current_focus_terms = self._provider_curriculum_focus_terms_locked(focus_plan)
        weak_concepts = [
            item
            for item in list((focus_plan or {}).get("weak_concepts") or [])
            if isinstance(item, Mapping)
        ]
        weak_focus_scale = 0.0
        if weak_concepts:
            weak_focus_scale = max(
                0.0,
                min(
                    1.0,
                    sum(
                        max(
                            0.0,
                            min(
                                1.0,
                                0.5 * float(item.get("weakness", 0.0))
                                + 0.5 * float(item.get("uncertainty", 0.0)),
                            ),
                        )
                        for item in weak_concepts
                    )
                    / float(len(weak_concepts)),
                ),
            )
        for raw_row in list(result.get("acquisition_history") or []):
            if not isinstance(raw_row, Mapping):
                continue
            provider = " ".join(str(raw_row.get("selected_provider", "")).split()).strip().lower()
            selected_metadata = raw_row.get("selected_metadata")
            if not provider and isinstance(selected_metadata, Mapping):
                provider = " ".join(str(selected_metadata.get("provider", "")).split()).strip().lower()
            entry = _ensure(provider)
            if not entry:
                continue
            entry["commits"] = int(entry.get("commits", 0)) + 1
            gap_gain = max(0.0, float(raw_row.get("selected_gap_reduction", 0.0)))
            diagnostic_gain = max(0.0, float(raw_row.get("selected_diagnostic_gap_reduction", 0.0)))
            semantic_relevance = max(0.0, min(1.0, float(raw_row.get("selected_semantic_relevance", 0.0))))
            selected_source = " ".join(str(raw_row.get("selected_source", "")).split()).strip()
            candidate_snapshot = raw_row.get("candidate_snapshot")
            before_metrics = {}
            if (
                selected_source
                and isinstance(candidate_snapshot, Mapping)
                and isinstance(candidate_snapshot.get(selected_source), Mapping)
            ):
                before_metrics = cast(Mapping[str, Any], candidate_snapshot.get(selected_source))
            answerability_before = max(
                0.0,
                min(1.0, float(before_metrics.get("semantic_answerability", 0.0) or 0.0)),
            )
            answerability_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_semantic_answerability_after", answerability_before) or answerability_before),
                ),
            )
            answerability_gain = max(0.0, answerability_after - answerability_before)
            uncertainty_before = max(
                0.0,
                min(1.0, float(before_metrics.get("concept_uncertainty", 0.0) or 0.0)),
            )
            uncertainty_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_concept_uncertainty_after", uncertainty_before) or uncertainty_before),
                ),
            )
            uncertainty_reduction = max(0.0, uncertainty_before - uncertainty_after)
            support_before = max(
                0.0,
                min(1.0, float(before_metrics.get("concept_support", 0.0) or 0.0)),
            )
            support_after = max(
                0.0,
                min(1.0, float(raw_row.get("selected_concept_support_after", support_before) or support_before)),
            )
            support_gain = max(0.0, support_after - support_before)
            weak_pressure_before = max(
                0.0,
                min(1.0, float(before_metrics.get("semantic_weak_concept_pressure", 0.0) or 0.0)),
            )
            weak_pressure_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_weak_concept_pressure_after", weak_pressure_before) or weak_pressure_before),
                ),
            )
            weak_pressure_reduction = max(0.0, weak_pressure_before - weak_pressure_after)
            weak_concept_stabilization = max(
                0.0,
                min(
                    1.0,
                    weak_focus_scale
                    * (
                        0.50 * uncertainty_reduction
                        + 0.30 * support_gain
                        + 0.20 * weak_pressure_reduction
                    ),
                ),
            )
            entry["gap_gain_ema"] = float(
                gap_gain
                if int(entry["commits"]) <= 1
                else 0.70 * float(entry.get("gap_gain_ema", 0.0)) + 0.30 * gap_gain
            )
            entry["diagnostic_gain_ema"] = float(
                diagnostic_gain
                if int(entry["commits"]) <= 1
                else 0.70 * float(entry.get("diagnostic_gain_ema", 0.0)) + 0.30 * diagnostic_gain
            )
            entry["semantic_relevance_ema"] = float(
                semantic_relevance
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
            )
            entry["answerability_gain_ema"] = float(
                answerability_gain
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
            )
            entry["uncertainty_reduction_ema"] = float(
                uncertainty_reduction
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("uncertainty_reduction_ema", 0.0)) + 0.25 * uncertainty_reduction
            )
            entry["weak_concept_stabilization_ema"] = float(
                weak_concept_stabilization
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("weak_concept_stabilization_ema", 0.0))
                + 0.25 * weak_concept_stabilization
            )
            if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                entry["successes"] = int(entry.get("successes", 0)) + 1
            query_text = " ".join(str(raw_row.get("selected_query_text", "")).split()).strip()
            if not query_text and isinstance(selected_metadata, Mapping):
                query_text = " ".join(str(selected_metadata.get("query_text", "")).split()).strip()
            if query_text:
                entry["last_query_text"] = query_text
            entry["last_selected_at"] = datetime.now(timezone.utc).isoformat()
            topic_terms = {
                str(term).strip().lower(): float(weight)
                for term, weight in dict(entry.get("topic_terms") or {}).items()
                if str(term).strip() and float(weight) > 0.0
            }
            for term in list(topic_terms):
                topic_terms[term] = float(topic_terms[term]) * 0.85
                if topic_terms[term] < 0.05:
                    topic_terms.pop(term, None)
            metadata_terms: list[str] = []
            if isinstance(selected_metadata, Mapping):
                metadata_terms = [
                    _canonical_provider_term(term)
                    for term in list(selected_metadata.get("catalog_terms") or [])
                    if _canonical_provider_term(term)
                ]
            update_terms = list(dict.fromkeys([*current_focus_terms, *metadata_terms]))
            if not update_terms and query_text:
                update_terms = [
                    normalized
                    for normalized in (_canonical_provider_term(term) for term in salient_query_terms(query_text))
                    if normalized
                ]
            for rank, term in enumerate(update_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]):
                topic_terms[term] = float(topic_terms.get(term, 0.0)) + 1.0 / float(rank + 1)
            entry["topic_terms"] = dict(
                sorted(
                    topic_terms.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            )
            topic_families: dict[str, dict[str, Any]] = {}
            for raw_family, raw_family_entry in dict(entry.get("topic_families") or {}).items():
                family = _canonical_provider_term(raw_family)
                if not family or not isinstance(raw_family_entry, Mapping):
                    continue
                topic_families[family] = {
                    "commits": max(0, int(raw_family_entry.get("commits", 0))),
                    "successes": max(0, int(raw_family_entry.get("successes", 0))),
                    "semantic_relevance_ema": max(
                        0.0,
                        float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                    ),
                    "answerability_gain_ema": max(
                        0.0,
                        float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                    ),
                    "uncertainty_reduction_ema": max(
                        0.0,
                        float(raw_family_entry.get("uncertainty_reduction_ema", 0.0)),
                    ),
                    "weak_concept_stabilization_ema": max(
                        0.0,
                        float(raw_family_entry.get("weak_concept_stabilization_ema", 0.0)),
                    ),
                    "last_selected_at": " ".join(str(raw_family_entry.get("last_selected_at", "")).split()).strip(),
                }
            for family_term in update_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]:
                topic_family = topic_families.setdefault(
                    family_term,
                    {
                        "commits": 0,
                        "successes": 0,
                        "semantic_relevance_ema": 0.0,
                        "answerability_gain_ema": 0.0,
                        "uncertainty_reduction_ema": 0.0,
                        "weak_concept_stabilization_ema": 0.0,
                        "last_selected_at": "",
                    },
                )
                topic_family["commits"] = int(topic_family.get("commits", 0)) + 1
                if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                    topic_family["successes"] = int(topic_family.get("successes", 0)) + 1
                topic_family["semantic_relevance_ema"] = float(
                    semantic_relevance
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
                )
                topic_family["answerability_gain_ema"] = float(
                    answerability_gain
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
                )
                topic_family["uncertainty_reduction_ema"] = float(
                    uncertainty_reduction
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("uncertainty_reduction_ema", 0.0))
                    + 0.25 * uncertainty_reduction
                )
                topic_family["weak_concept_stabilization_ema"] = float(
                    weak_concept_stabilization
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("weak_concept_stabilization_ema", 0.0))
                    + 0.25 * weak_concept_stabilization
                )
                topic_family["last_selected_at"] = entry["last_selected_at"]
            entry["topic_families"] = dict(
                sorted(
                    topic_families.items(),
                    key=lambda item: (-self._provider_topic_family_priority_locked(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            )
            query_families: dict[str, dict[str, Any]] = {}
            for raw_family, raw_family_entry in dict(entry.get("query_families") or {}).items():
                family = _canonical_provider_term(raw_family)
                if not family or not isinstance(raw_family_entry, Mapping):
                    continue
                query_families[family] = {
                    "commits": max(0, int(raw_family_entry.get("commits", 0))),
                    "successes": max(0, int(raw_family_entry.get("successes", 0))),
                    "semantic_relevance_ema": max(
                        0.0,
                        float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                    ),
                    "answerability_gain_ema": max(
                        0.0,
                        float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                    ),
                    "uncertainty_reduction_ema": max(
                        0.0,
                        float(raw_family_entry.get("uncertainty_reduction_ema", 0.0)),
                    ),
                    "weak_concept_stabilization_ema": max(
                        0.0,
                        float(raw_family_entry.get("weak_concept_stabilization_ema", 0.0)),
                    ),
                    "last_selected_at": " ".join(str(raw_family_entry.get("last_selected_at", "")).split()).strip(),
                }
            geometric_queries = [
                " ".join(str(query).split()).strip().lower()
                for query in list((focus_plan or {}).get("retrieval_queries") or [])
                if " ".join(str(query).split()).strip()
            ]
            update_query_families = list(dict.fromkeys([query_text.lower(), *geometric_queries])) if query_text else list(dict.fromkeys(geometric_queries))
            if not update_query_families and query_text:
                update_query_families = [query_text.lower()]
            for family_query in update_query_families[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]:
                query_family = query_families.setdefault(
                    family_query,
                    {
                        "commits": 0,
                        "successes": 0,
                        "semantic_relevance_ema": 0.0,
                        "answerability_gain_ema": 0.0,
                        "uncertainty_reduction_ema": 0.0,
                        "weak_concept_stabilization_ema": 0.0,
                        "last_selected_at": "",
                    },
                )
                query_family["commits"] = int(query_family.get("commits", 0)) + 1
                if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                    query_family["successes"] = int(query_family.get("successes", 0)) + 1
                query_family["semantic_relevance_ema"] = float(
                    semantic_relevance
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
                )
                query_family["answerability_gain_ema"] = float(
                    answerability_gain
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
                )
                query_family["uncertainty_reduction_ema"] = float(
                    uncertainty_reduction
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("uncertainty_reduction_ema", 0.0))
                    + 0.25 * uncertainty_reduction
                )
                query_family["weak_concept_stabilization_ema"] = float(
                    weak_concept_stabilization
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("weak_concept_stabilization_ema", 0.0))
                    + 0.25 * weak_concept_stabilization
                )
                query_family["last_selected_at"] = entry["last_selected_at"]
            entry["query_families"] = dict(
                sorted(
                    query_families.items(),
                    key=lambda item: (-self._provider_query_family_priority_locked(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
            )

        autonomy["provider_curriculum"] = curriculum

    def _candidate_pool_size_hint(self, candidate_bank: Sequence[dict[str, Any]]) -> int:
        estimated_pool_size = 0
        for spec in candidate_bank:
            if not isinstance(spec, dict):
                continue
            catalog_mode = str(spec.get("catalog_mode", "")).strip().lower()
            if not catalog_mode:
                estimated_pool_size += 1
                continue
            catalog_entries = spec.get("catalog_entries")
            entry_count = 0
            if isinstance(catalog_entries, Sequence) and not isinstance(catalog_entries, (str, bytes)):
                entry_count = len(list(catalog_entries))
            catalog_limit = max(1, int(spec.get("catalog_limit", max(1, entry_count or 1))))
            probe_pool_limit = int(spec.get("catalog_probe_pool_limit", 0) or 0)
            if probe_pool_limit > 0:
                estimated_pool_size += max(catalog_limit, probe_pool_limit, entry_count)
                continue
            if catalog_mode == "live_remote_search":
                provider_count = max(
                    1,
                    len(
                        [
                            str(item).strip()
                            for item in list(spec.get("catalog_providers") or [])
                            if str(item).strip()
                        ]
                    ),
                )
                query_count = max(1, int(spec.get("catalog_queries_per_provider", 2)))
                result_limit = max(1, int(spec.get("catalog_provider_result_limit", catalog_limit)))
                estimated_pool_size += max(catalog_limit, provider_count * query_count * result_limit)
                continue
            estimated_pool_size += max(catalog_limit, entry_count)
        return estimated_pool_size

    def _autonomy_shortlist_settings_locked(
        self,
        *,
        candidate_bank: list[dict[str, Any]],
        config: dict[str, Any],
        focus_plan: dict[str, Any] | None,
    ) -> tuple[int, float, float]:
        shortlist_size = max(0, int(config.get("semantic_shortlist_size", 0)))
        gap_weight = float(config.get("semantic_shortlist_gap_weight", 0.5))
        affinity_weight = float(config.get("semantic_shortlist_affinity_weight", 0.5))
        if shortlist_size > 0:
            return shortlist_size, gap_weight, affinity_weight
        if focus_plan is None:
            return shortlist_size, gap_weight, affinity_weight

        focus_signal_count = int(len(list(focus_plan.get("unsupported_terms") or [])))
        focus_signal_count += int(len(list(focus_plan.get("retrieval_queries") or [])))
        focus_signal_count += int(len(list(focus_plan.get("gap_terms") or [])))
        focus_signal_count += int(len(list(focus_plan.get("weak_concepts") or [])))
        if focus_signal_count <= 0:
            return shortlist_size, gap_weight, affinity_weight

        estimated_pool_size = self._candidate_pool_size_hint(candidate_bank)
        if estimated_pool_size <= 1:
            return shortlist_size, gap_weight, affinity_weight
        auto_size = max(1, min(AUTO_FOCUS_SHORTLIST_MAX_SIZE, (estimated_pool_size + 1) // 2))
        return auto_size, AUTO_FOCUS_SHORTLIST_GAP_WEIGHT, AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT

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
