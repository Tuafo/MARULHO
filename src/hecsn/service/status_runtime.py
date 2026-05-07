from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Any, Mapping

import torch

from hecsn.data.corpus_loader import huggingface_token_from_env

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_REPLAY_DATASET_EXPORT_LIMIT = 20


class StatusRuntimeMixin:
    """Status, telemetry, and runtime warm-state summaries."""

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

    @staticmethod
    def _replay_dataset_summary_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any] | None:
        living_loop = runtime.get("living_loop") if isinstance(runtime, Mapping) else None
        if not isinstance(living_loop, Mapping):
            return None
        summary = living_loop.get("replay_dataset_summary")
        return deepcopy(dict(summary)) if isinstance(summary, Mapping) else None

    @staticmethod
    def _runtime_source_configuration_evidence(terminus_runtime: Mapping[str, Any]) -> dict[str, Any]:
        source_bank = [
            deepcopy(dict(item))
            for item in list(terminus_runtime.get("source_bank") or [])
            if isinstance(item, Mapping)
        ]
        sensory = terminus_runtime.get("sensory") if isinstance(terminus_runtime.get("sensory"), Mapping) else {}
        sensory_source_bank = [
            deepcopy(dict(item))
            for item in list(sensory.get("source_bank") or [])
            if isinstance(item, Mapping)
        ]
        ingestion = terminus_runtime.get("ingestion") if isinstance(terminus_runtime.get("ingestion"), Mapping) else {}
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
                else "watch_working_set_growth" if pressure == "medium" else "continue_monitoring"
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

    def _status_snapshot_locked(self) -> dict[str, Any]:
        last_trace = self._trace_history[0] if self._trace_history else None
        terminus_runtime = self._brain_runtime_snapshot_locked()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size = int(len(self._trace_history))
        return {
            "checkpoint_path": str(self._checkpoint_path),
            "dirty_state": bool(self._runtime_state.dirty_state),
            "state_revision": int(self._runtime_state.state_revision),
            "token_count": int(self._trainer.token_count),
            "last_winner": None if self._trainer.last_winner is None else int(self._trainer.last_winner),
            "context_supported": bool(self._trainer.model.context_layer is not None),
            "context_state_norm": float(torch.norm(self._trainer.context_state().float()).item()),
            "trace_history_size": trace_history_size,
            "trace_storage_dir": str(self._trace_dir),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "last_trace_created_at": None if last_trace is None else str(last_trace.get("created_at")),
            "checkpoint_metadata": deepcopy(self._metadata),
            "dopamine": float(self._trainer.model.surprise.dopamine),
            "serotonin": float(self._trainer.model.surprise.serotonin),
            "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
            "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
            "runtime_scope": self._trainer.model.runtime_scope_report(),
            "memory_store": memory_store,
            "concept_store": self._concept_store.snapshot(),
            "terminus_runtime": terminus_runtime,
            "replay_dataset_summary": replay_dataset_summary,
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
            ),
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
        current_rev = int(self._runtime_state.state_revision)
        cortex_active = self._thought_loop_actual is not None and self._thought_loop_actual.is_running
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
        terminus_runtime = self._brain_runtime_snapshot_locked()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(self._checkpoint_path),
            "dirty_state": bool(self._runtime_state.dirty_state),
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
            "terminus_runtime": terminus_runtime,
            "replay_dataset_summary": replay_dataset_summary,
        }
        self._cached_telemetry = snapshot
        self._cached_telemetry_rev = current_rev
        return snapshot

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
        terminus_runtime = self._brain_runtime_snapshot_locked()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size = int(len(self._trace_history))
        return {
            "terminus_runtime": terminus_runtime,
            "dirty_state": bool(self._runtime_state.dirty_state),
            "state_revision": int(self._runtime_state.state_revision),
            "token_count": int(self._trainer.token_count),
            "multimodal": self._multimodal_runtime_summary_locked(),
            "replay_dataset_summary": replay_dataset_summary,
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
            ),
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

