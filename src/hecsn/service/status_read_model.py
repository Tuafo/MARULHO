"""Status Read Model — read-only projection of runtime state into status and
terminus snapshots.

This module owns the ``status()`` and ``terminus_status()`` public surfaces
and their associated cache state.  It is the first real object-style
extraction from ADR 0003: the Service Manager delegates to it, and direct
object-level tests exercise the seam through injected adapter callbacks
instead of requiring the full manager composition root.

The read model is strictly read-only: it never mutates RuntimeState, never
records brain events, and never advances the revision counter.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Mapping
import hashlib
import json
import time

import torch

from hecsn.data.corpus_loader import huggingface_token_from_env
from hecsn.service.runtime_state import RuntimeState

DEFAULT_BRAIN_TICK_TOKENS = 512


class StatusReadModel:
    """Read-only projection of runtime state for ``status()`` and
    ``terminus_status()``.

    Dependencies are injected at construction so that the read model can
    be exercised in isolation with fakes, while the production wiring in
    the Service Manager passes callbacks that read from the real manager
    state under the shared lock.

    Parameters
    ----------
    lock:
        The shared ``RLock`` used by the Service Manager and all deep
        modules.  The read model uses it for thread-safe cache access
        but never owns it.
    runtime_state:
        The ``RuntimeState`` deep module for mutation summary reads.
    trainer:
        The ``HECSNTrainer`` instance (read-only access to model stats).
    trace_history:
        The manager's trace history deque (read-only access).
    metadata:
        The manager's checkpoint metadata dict (read-only access).
    checkpoint_path_str:
        String form of the checkpoint path for payload inclusion.
    trace_dir_str:
        String form of the trace directory for payload inclusion.
    concept_store_snapshot_fn:
        Callable that returns the current concept store snapshot dict.
        Called under lock.
    brain_runtime_snapshot_fn:
        Callable that returns the current brain runtime snapshot dict.
        Called under lock.
    multimodal_runtime_summary_fn:
        Callable that returns the current multimodal runtime summary
        dict.  Called under lock.  Used by ``terminus_status()`` only.
    """

    def __init__(
        self,
        *,
        lock: RLock,
        runtime_state: RuntimeState,
        trainer: Any,
        trace_history: deque[dict[str, Any]],
        metadata: dict[str, Any],
        checkpoint_path_str: str,
        trace_dir_str: str,
        concept_store_snapshot_fn: Callable[[], dict[str, Any]],
        brain_runtime_snapshot_fn: Callable[[], dict[str, Any]],
        multimodal_runtime_summary_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._trainer = trainer
        self._trace_history = trace_history
        self._metadata = metadata
        self._checkpoint_path_str = checkpoint_path_str
        self._trace_dir_str = trace_dir_str
        self._concept_store_snapshot_fn = concept_store_snapshot_fn
        self._brain_runtime_snapshot_fn = brain_runtime_snapshot_fn
        self._multimodal_runtime_summary_fn = multimodal_runtime_summary_fn
        # Cache state — owned by the read model
        self._cached_status: dict[str, Any] | None = None
        self._cached_terminus_status: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Static helpers reused from StatusRuntimeMixin
    # ------------------------------------------------------------------

    @staticmethod
    def _replay_dataset_summary_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any] | None:
        """Extract replay dataset summary from a terminus runtime snapshot."""
        living_loop = runtime.get("living_loop") if isinstance(runtime, Mapping) else None
        if not isinstance(living_loop, Mapping):
            return None
        summary = living_loop.get("replay_dataset_summary")
        return deepcopy(dict(summary)) if isinstance(summary, Mapping) else None

    @staticmethod
    def _runtime_source_configuration_evidence(terminus_runtime: Mapping[str, Any]) -> dict[str, Any]:
        """Build source configuration evidence from a terminus runtime snapshot."""
        source_bank = [
            deepcopy(dict(item))
            for item in list(terminus_runtime.get("source_bank") or [])
            if isinstance(item, Mapping)
        ]
        sensory_raw = terminus_runtime.get("sensory")
        sensory: Mapping[str, Any] = sensory_raw if isinstance(sensory_raw, Mapping) else {}
        sensory_source_bank = [
            deepcopy(dict(item))
            for item in list(sensory.get("source_bank") or [])
            if isinstance(item, Mapping)
        ]
        ingestion_raw = terminus_runtime.get("ingestion")
        ingestion: Mapping[str, Any] = ingestion_raw if isinstance(ingestion_raw, Mapping) else {}
        payload = {
            "source_bank": source_bank,
            "sensory_source_bank": sensory_source_bank,
            "tick_tokens": int(terminus_runtime.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS) or DEFAULT_BRAIN_TICK_TOKENS),
            "sleep_interval_seconds": float(terminus_runtime.get("sleep_interval_seconds", 0.0) or 0.0),
            "repeat_sources": bool(terminus_runtime.get("repeat_sources", True)),
            "ingestion": {
                "enabled": bool(ingestion.get("enabled", True)),
                "queue_target_tokens": int(ingestion.get("queue_target_tokens", 0) or 0),
                "prewarm_on_startup": bool(ingestion.get("prewarm_on_startup", False)),
                "prewarm_max_seconds": float(ingestion.get("prewarm_max_seconds", 0.0) or 0.0),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return {
            "configured": bool(terminus_runtime.get("configured")),
            "source_count": int(len(source_bank)),
            "source_names": [str(item.get("name", "")) for item in source_bank],
            "source_types": [str(item.get("source_type", "auto")) for item in source_bank],
            "sensory_source_count": int(len(sensory_source_bank)),
            "sensory_source_names": [str(item.get("name", "")) for item in sensory_source_bank],
            "tick_tokens": payload["tick_tokens"],
            "sleep_interval_seconds": payload["sleep_interval_seconds"],
            "repeat_sources": payload["repeat_sources"],
            "ingestion": payload["ingestion"],
            "configuration_hash": hashlib.sha256(encoded).hexdigest(),
            "configuration_payload": payload,
            "operator_action": (
                "configure_terminus_sources"
                if not bool(terminus_runtime.get("configured"))
                else "use_configuration_hash_and_payload_to_reproduce_run"
            ),
        }

    def _runtime_truth_contract_locked(
        self,
        *,
        terminus_runtime: Mapping[str, Any],
        memory_store: Mapping[str, Any],
        replay_dataset_summary: Mapping[str, Any] | None,
        trace_history_size: int,
    ) -> dict[str, Any]:
        """Build the Runtime Truth contract. Reads self._trainer for token_count only."""
        cortex = terminus_runtime.get("cortex") if isinstance(terminus_runtime, Mapping) else {}
        cortex_available = bool(isinstance(cortex, Mapping) and cortex.get("enabled"))
        configured = bool(terminus_runtime.get("configured"))
        running = bool(terminus_runtime.get("running"))
        last_error = str(terminus_runtime.get("last_error") or "").strip()
        tick_count = max(0, int(terminus_runtime.get("tick_count", 0) or 0))
        background_tokens = max(0, int(terminus_runtime.get("background_tokens_processed", 0) or 0))
        autonomy_tokens = max(0, int(terminus_runtime.get("autonomy_tokens_processed", 0) or 0))
        token_count = max(0, int(self._trainer.token_count))
        last_work_at = terminus_runtime.get("last_work_at")
        progress_observed = bool(
            running
            or tick_count > 0
            or background_tokens > 0
            or autonomy_tokens > 0
            or token_count > 0
            or trace_history_size > 0
            or last_work_at
        )
        if last_error:
            verdict = "failed"
            recommended_action = "inspect_last_error"
        elif not configured:
            verdict = "partial"
            recommended_action = "configure_terminus_sources"
        elif not cortex_available:
            verdict = "partial"
            recommended_action = "initialize_or_configure_cortex"
        elif not progress_observed:
            verdict = "degraded"
            recommended_action = "run_tick_or_start_runtime"
        else:
            verdict = "alive"
            recommended_action = "continue_monitoring"

        replay_role = "none"
        replay_endpoint = None
        replay_safety_flags: dict[str, Any] = {}
        if isinstance(replay_dataset_summary, Mapping):
            replay_endpoint = replay_dataset_summary.get("endpoint")
            replay_safety_flags = dict(replay_dataset_summary.get("safety_flags") or {})
            replay_role = str(replay_dataset_summary.get("training_role") or "preview_export_only")

        fill_fraction = float(memory_store.get("fill_fraction", 0.0) or 0.0)
        pressure = "high" if fill_fraction >= 0.85 else "medium" if fill_fraction >= 0.50 else "low"

        working_set_policy = {
            "high_threshold": 0.85,
            "target_fill": 0.70,
            "capacity_increase_recommended": False,
            "replay_fact_promotion_allowed": False,
            "decision": (
                "throttle_ingestion_and_prioritize_consolidation"
                if pressure == "high"
                else "watch_working_set_growth"
                if pressure == "medium"
                else "continue_monitoring"
            ),
        }
        if verdict == "alive" and pressure == "high":
            verdict = "degraded"
            recommended_action = "reduce_memory_pressure_before_extending_runtime"

        memory_pressure = {
            "fill_fraction": fill_fraction,
            "size": int(memory_store.get("size", 0) or 0),
            "capacity": int(memory_store.get("capacity", 0) or 0),
            "pressure": pressure,
            "working_set_policy": working_set_policy,
        }

        last_tick_duration = terminus_runtime.get("last_tick_duration_ms")
        latency_ms = {
            "last_tick": None if last_tick_duration is None else float(last_tick_duration),
            "tokens_per_second": float(terminus_runtime.get("tokens_per_second", 0.0) or 0.0),
        }

        source_configuration = self._runtime_source_configuration_evidence(terminus_runtime)

        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "recommended_action": recommended_action,
            "cortex_available": cortex_available,
            "source_configuration": source_configuration,
            "memory_pressure": memory_pressure,
            "replay_role": replay_role,
            "safety_flags": {
                "replay_dataset_preview_only": replay_role != "training",
                "replay_safety": replay_safety_flags,
            },
            "latency_ms": latency_ms,
            "evidence": {
                "configured": configured,
                "running": running,
                "token_count": token_count,
                "trace_history_size": int(trace_history_size),
                "tick_count": tick_count,
                "background_tokens_processed": background_tokens,
                "autonomy_tokens_processed": autonomy_tokens,
                "last_work_at": last_work_at,
                "last_error": last_error or None,
                "cortex_enabled": cortex_available,
                "replay_endpoint": replay_endpoint,
                "source_configuration_hash": source_configuration["configuration_hash"],
            },
        }

    # ------------------------------------------------------------------
    # Internal snapshot builders (must be called under self._lock)
    # ------------------------------------------------------------------

    def _status_snapshot_locked(self) -> dict[str, Any]:
        """Build the full status snapshot. Caller MUST hold self._lock."""
        last_trace = self._trace_history[0] if self._trace_history else None
        terminus_runtime = self._brain_runtime_snapshot_fn()
        runtime_mutation = self._runtime_state.mutation_summary()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size = int(len(self._trace_history))
        return {
            "checkpoint_path": str(self._checkpoint_path_str),
            **runtime_mutation,
            "token_count": int(self._trainer.token_count),
            "last_winner": (
                None if self._trainer.last_winner is None
                else int(self._trainer.last_winner)
            ),
            "context_supported": bool(self._trainer.model.context_layer is not None),
            "context_state_norm": float(
                torch.linalg.norm(self._trainer.context_state().float()).item()
            ),
            "trace_history_size": trace_history_size,
            "trace_storage_dir": str(self._trace_dir_str),
            "last_trace_id": (
                None if last_trace is None
                else str(last_trace.get("trace_id"))
            ),
            "last_trace_created_at": (
                None if last_trace is None
                else str(last_trace.get("created_at"))
            ),
            "checkpoint_metadata": deepcopy(self._metadata),
            "dopamine": float(self._trainer.model.surprise.dopamine),
            "serotonin": float(self._trainer.model.surprise.serotonin),
            "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
            "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
            "runtime_scope": self._trainer.model.runtime_scope_report(),
            "memory_store": memory_store,
            "concept_store": self._concept_store_snapshot_fn(),
            "terminus_runtime": terminus_runtime,
            "replay_dataset_summary": replay_dataset_summary,
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
            ),
        }

    def _terminus_status_snapshot_locked(self) -> dict[str, Any]:
        """Build the terminus status snapshot. Caller MUST hold self._lock."""
        terminus_runtime = self._brain_runtime_snapshot_fn()
        runtime_mutation = self._runtime_state.mutation_summary()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size = int(len(self._trace_history))
        multimodal = (
            self._multimodal_runtime_summary_fn()
            if self._multimodal_runtime_summary_fn is not None
            else {}
        )
        return {
            "terminus_runtime": terminus_runtime,
            **runtime_mutation,
            "token_count": int(self._trainer.token_count),
            "multimodal": multimodal,
            "replay_dataset_summary": replay_dataset_summary,
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
            ),
        }

    # ------------------------------------------------------------------
    # Public surfaces
    # ------------------------------------------------------------------

    def status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Return the status snapshot (non-blocking by default).

        When ``fresh_wait_seconds`` is ``None`` the call tries to acquire
        the lock for up to 150 ms.  If that fails and a cached snapshot
        exists, the cached copy is returned.  When a caller explicitly
        requests a fresh snapshot, the method retries until the deadline
        and then blocks rather than silently serving stale data.
        """
        if fresh_wait_seconds is None:
            acquired = self._lock.acquire(timeout=0.15)
            if not acquired:
                cached = self._cached_status
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

    def terminus_status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Return the terminus status snapshot (non-blocking by default).

        Polling semantics mirror ``status()``: non-blocking default with
        cached fallback, and blocking retry when an explicit freshness
        deadline is requested.
        """
        if fresh_wait_seconds is None:
            acquired = self._lock.acquire(timeout=0.15)
            if not acquired:
                cached = self._cached_terminus_status
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
