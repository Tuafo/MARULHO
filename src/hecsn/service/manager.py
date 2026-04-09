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
from hecsn.data.corpus_loader import StreamingCorpusLoader
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.gap_planner import plan_query_gaps
from hecsn.interaction import EvidenceResponder
from hecsn.reporting.io import write_json_file
from hecsn.semantics import ConceptStore
from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.training.autonomy_acquisition_runner import run_live_acquisition
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.query_runner import build_query_result, feed_text


PUBLIC_ACQUISITION_PRESET = "autonomy_acquisition_hf_allocation"
PUBLIC_ACQUISITION_PRESETS: tuple[str, ...] = (PUBLIC_ACQUISITION_PRESET,)
PUBLIC_ACQUISITION_POLICIES: tuple[str, ...] = ("active", "round_robin")
DEFAULT_BRAIN_TICK_TOKENS = 128
DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS = 0.25
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
AUTO_FOCUS_SHORTLIST_MAX_SIZE = 3
AUTO_FOCUS_SHORTLIST_GAP_WEIGHT = 0.2
AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT = 0.8


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
        self._encoder = RTFEncoder.from_config(self._trainer.config)
        self._responder = EvidenceResponder()
        self._trace_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(trace_history_limit)))
        service_state = dict(self._metadata.get("service_state", {}))
        terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
        concept_state = service_state.get("concept_store")
        self._concept_store = ConceptStore.from_state_dict(concept_state)
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
        self._rebuild_brain_sources_locked()
        self._dirty_state = False
        self._state_revision = 0
        self._load_persisted_traces_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
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
                "runtime_scope": self._trainer.model.runtime_scope_report(),
                "memory_store": self._trainer.model.memory_store.summary_stats(),
                "concept_store": self._concept_store.snapshot(),
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
            }

    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._lock:
            memory_store = self._trainer.model.memory_store.summary_stats()
            last_trace = self._trace_history[0] if self._trace_history else None
            drift_bucket = self._trainer.last_winner if self._trainer.config.use_winner_local_drift else None
            drift_value = self._trainer.model.memory_store.compute_drift(drift_bucket)
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(self._checkpoint_path),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
                "last_winner": None if self._trainer.last_winner is None else int(self._trainer.last_winner),
                "context_state_norm": float(torch.norm(self._trainer.context_state().float()).item()),
                "trace_history_size": int(len(self._trace_history)),
                "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
                "last_trace_created_at": None if last_trace is None else str(last_trace.get("created_at")),
                "memory_fill_fraction": float(memory_store.get("slow_buffer_fill_fraction", 0.0)),
                "memory_buffer_size": int(memory_store.get("slow_buffer_size", 0)),
                "sleep_events": int(self._trainer.sleep_events),
                "micro_sleep_events": int(self._trainer.micro_sleep_events),
                "deep_sleep_events": int(self._trainer.deep_sleep_events),
                "dopamine": float(self._trainer.model.surprise.dopamine),
                "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
                "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
                "drift": float(drift_value),
                "drift_floor": float(self._trainer.current_rolling_drift_floor if self._trainer.current_rolling_drift_floor is not None else drift_value),
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
            }

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
        with self._lock:
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def configure_terminus(
        self,
        *,
        source_bank: list[dict[str, Any]],
        tick_tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        sleep_interval_seconds: float = DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
        repeat_sources: bool = True,
        autonomy: dict[str, Any] | None = None,
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
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "dirty_state": bool(self._dirty_state),
                "state_revision": int(self._state_revision),
                "token_count": int(self._trainer.token_count),
            }

    def stop_terminus(self) -> dict[str, Any]:
        thread = self._request_brain_stop(reason="manual")
        self._join_brain_thread(thread)
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

    def close(self) -> None:
        thread = self._request_brain_stop(reason="shutdown")
        self._join_brain_thread(thread)
        with self._lock:
            self._close_brain_sources_locked()

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
            self._encoder = RTFEncoder.from_config(self._trainer.config)
            self._checkpoint_path = checkpoint_path
            self._checkpoint_dir = checkpoint_path.parent if checkpoint_path.parent != Path("") else Path("checkpoints")
            service_state = dict(self._metadata.get("service_state", {}))
            terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
            concept_state = service_state.get("concept_store")
            self._concept_store = ConceptStore.from_state_dict(concept_state)
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

        return self._concept_store.observe(
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
                    term = " ".join(str(raw_term).split()).strip().lower()
                    if not term:
                        continue
                    weight = _safe_float(raw_weight)
                    if weight > 0.0:
                        topic_terms[term] = float(weight)
            topic_families: dict[str, dict[str, Any]] = {}
            raw_topic_families = raw_entry.get("topic_families")
            if isinstance(raw_topic_families, Mapping):
                for raw_family, raw_family_entry in raw_topic_families.items():
                    family = " ".join(str(raw_family).split()).strip().lower()
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
        }
        return normalized

    def _build_brain_source_stream_locked(self, spec: dict[str, Any]) -> Iterator[tuple[str, torch.Tensor]]:
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=str(spec.get("source_type", "auto")),
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        return labeled_pattern_stream(loader.char_stream(), self._encoder, self._trainer.config.window_size)

    def _close_brain_sources_locked(self) -> None:
        for runtime in self._brain_source_runtimes:
            close = getattr(runtime.stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue
        self._brain_source_runtimes = []

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
        while True:
            with self._lock:
                stop_event = self._brain_stop_event
                sleep_interval = float(self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS))
            if stop_event is None or stop_event.is_set():
                break
            try:
                with self._lock:
                    self._brain_tick_locked()
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
            time.sleep(max(0.01, sleep_interval))

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
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None
        token_count_after = int(self._trainer.token_count)
        completed_at = datetime.now(timezone.utc).isoformat()
        token_delta = int(token_count_after - token_count_before)
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": source_summary,
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
        if token_delta < trigger_interval:
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
            },
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
        meaningful = bool(gap_plan.get("unsupported_terms") or gap_plan.get("gap_terms") or gap_plan.get("weak_concepts"))
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
        if recent_query_focus is None:
            return self._concept_store.focus_plan()
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
        concept_focus = self._concept_store.focus_plan(
            query_text=abstraction_query,
            min_observations=1,
        )
        if concept_focus is None:
            return recent_query_focus
        return self._merge_focus_plans_locked(recent_query_focus, concept_focus)

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

        for plan, plan_weight in ((primary, 1.0), (secondary, 0.75)):
            query_terms.extend(str(term) for term in list(plan.get("query_terms") or []) if str(term).strip())
            retrieval_queries.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
            follow_up_questions.extend(str(item) for item in list(plan.get("follow_up_questions") or []) if str(item).strip())

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
                *[str(item) for item in list(focus_plan.get("retrieval_queries") or [])[:2]],
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
                    normalized = " ".join(str(term).split()).strip().lower()
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
        normalized_family = " ".join(str(family_term).split()).strip().lower()
        if not normalized_family:
            return 0.0
        family_tokens = {term.lower() for term in salient_query_terms(normalized_family)}
        if not family_tokens:
            family_tokens = {part for part in normalized_family.split() if part}
        if not family_tokens:
            return 0.0
        best = 0.0
        for raw_focus in focus_terms:
            normalized_focus = " ".join(str(raw_focus).split()).strip().lower()
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
            "last_query_text": str(entry.get("last_query_text", "")),
            "last_selected_at": str(entry.get("last_selected_at", "")),
            "topic_terms": dict(entry.get("topic_terms") or {}),
            "topic_families": dict(entry.get("topic_families") or {}),
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
        topic_family_query_bonus = 0
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
        ranked.sort(key=lambda item: (-float(item[2]), int(item[0])))
        spec["catalog_providers"] = [provider for _index, provider, _priority in ranked]
        spec["catalog_provider_priority_map"] = dict(priority_map)
        if provider_topic_terms:
            spec["catalog_provider_topic_terms"] = dict(provider_topic_terms)
        spec["catalog_topic_family_budget_bonus"] = int(topic_family_query_bonus)
        if topic_family_query_bonus > 0:
            spec["catalog_queries_per_provider"] = int(
                min(
                    AUTO_REMOTE_QUERY_BUDGET_MAX,
                    max(1, int(spec.get("catalog_queries_per_provider", 1))) + topic_family_query_bonus,
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
                },
            )
            entry["topic_terms"] = dict(entry.get("topic_terms") or {})
            entry["topic_families"] = dict(entry.get("topic_families") or {})
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
                    str(term).strip().lower()
                    for term in list(selected_metadata.get("catalog_terms") or [])
                    if str(term).strip()
                ]
            update_terms = list(dict.fromkeys([*current_focus_terms, *metadata_terms]))
            if not update_terms and query_text:
                update_terms = [term.lower() for term in salient_query_terms(query_text)]
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
                family = " ".join(str(raw_family).split()).strip().lower()
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
