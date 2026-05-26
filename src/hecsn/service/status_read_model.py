"""Status Read Model — read-only projection of runtime state into status, terminus, telemetry, living-loop, policy-actuator, and cognitive-signal snapshots.

This module owns the ``status()``, ``terminus_status()``, ``telemetry_snapshot()``,
``living_loop_status()``, ``policy_actuator_status()``, and ``cognitive_signal_state()``
public surfaces and their associated cache state. It
is the first real object-style extraction from ADR 0003: the Service Manager
delegates to it, and direct object-level tests exercise the seam through
injected adapter callbacks instead of requiring the full manager composition
root.

The read model is strictly read-only: it never mutates RuntimeState, never
records brain events, and never advances the revision counter.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Mapping
import base64
import hashlib
import json
import time

import torch

from hecsn.semantics import (
    attach_cognitive_signal_language_surface,
    build_snn_language_readiness_surface,
    build_subcortical_self_repair_evaluation_surface,
    build_subcortical_self_repair_surface,
    build_subcortical_structural_plasticity_surface,
)
from hecsn.service.runtime_state import RuntimeState

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS = 0.15
DEFAULT_COGNITIVE_SIGNAL_LOCK_TIMEOUT_SECONDS = 0.05
DEFAULT_CORTEX_SIGNAL_LOCK_TIMEOUT_SECONDS = DEFAULT_COGNITIVE_SIGNAL_LOCK_TIMEOUT_SECONDS


def _default_architecture_snapshot() -> dict[str, Any]:
    return {
        "model_name": "Terminus",
        "core_name": "GPCSN",
        "version": "current",
        "family": "subcortex_runtime",
        "layers": [],
        "config": {},
    }


class StatusReadModel:
    """Read-only projection of runtime state for ``status()``, ``terminus_status()``,
    ``sensory_previews()``, ``architecture_summary()``, and ``telemetry_snapshot()``.

    Dependencies are injected at construction so that the read model can be
    exercised in isolation with fakes, while the production wiring in the
    Service Manager passes callbacks that read from the real manager state
    under the shared lock.

    Parameters
    ----------
    lock: The shared ``RLock`` used by the Service Manager and all deep modules.
        The read model uses it for thread-safe cache access but never owns it.
    runtime_state: The ``RuntimeState`` deep module for mutation summary reads.
    trainer: The ``HECSNTrainer`` instance (read-only access to model stats).
    trace_history: The manager's trace history deque (read-only access).
    metadata: The manager's checkpoint metadata dict (read-only access).
    checkpoint_path_str: String form of the checkpoint path for payload inclusion.
    trace_dir_str: String form of the trace directory for payload inclusion.
    concept_store_snapshot_fn: Callable that returns the current concept store
        snapshot dict. Called under lock.
    brain_runtime_snapshot_fn: Callable that returns the current brain runtime
        snapshot dict. Called under lock.
    multimodal_runtime_summary_fn: Callable that returns the current multimodal
        runtime summary dict. Called under lock. Used by ``terminus_status()``
        only.
    cortex_active_fn: Callable that returns whether the cortex thought loop is
        currently running. Called under lock. Used by ``telemetry_snapshot()``
        for revision-keyed cache reuse decisions.
    animation_snapshot_fn: Callable that returns the current animation snapshot
        dict. Called under lock. Used by ``telemetry_snapshot()`` only.
    living_loop_status_fn: Callable that returns the current living loop status
        dict. Called under lock. Used by ``living_loop_status()`` delegation.
    policy_actuator_status_fn: Callable that returns the current policy actuator
        status dict. Called under lock. Used by ``policy_actuator_status()``
        delegation.
    cognitive_signal_state_fn: Callable that returns the current Cognitive Signal
        dict. Called under lock with short-timeout fallback. Used by
        ``cognitive_signal_state()`` delegation.
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
        sensory_preview_history: deque[dict[str, Any]] | None = None,
        architecture_snapshot_fn: Callable[[], dict[str, Any]] | None = None,
        cortex_active_fn: Callable[[], bool] | None = None,
        animation_snapshot_fn: Callable[[], dict[str, Any]] | None = None,
        living_loop_status_fn: Callable[[], dict[str, Any]] | None = None,
        policy_actuator_status_fn: Callable[[], dict[str, Any]] | None = None,
        cognitive_signal_state_fn: Callable[[], dict[str, Any]] | None = None,
        cortex_signal_state_fn: Callable[[], dict[str, Any]] | None = None,
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
        self._sensory_preview_history = sensory_preview_history if sensory_preview_history is not None else deque(maxlen=8)
        self._architecture_snapshot_fn = (
            architecture_snapshot_fn
            if architecture_snapshot_fn is not None
            else _default_architecture_snapshot
        )
        self._cortex_active_fn = cortex_active_fn
        self._animation_snapshot_fn = animation_snapshot_fn
        self._living_loop_status_fn = living_loop_status_fn
        self._policy_actuator_status_fn = policy_actuator_status_fn
        self._cognitive_signal_state_fn = cognitive_signal_state_fn or cortex_signal_state_fn

        # Cache state — owned by the read model
        self._cached_status: dict[str, Any] | None = None
        self._cached_terminus_status: dict[str, Any] | None = None
        self._cached_telemetry: dict[str, Any] | None = None
        self._cached_telemetry_rev: int = -1
        self._cached_cognitive_signal_state: dict[str, Any] | None = None
        self._cached_cortex_signal_state: dict[str, Any] | None = None
        self._cached_snn_language_readiness_surface: dict[str, Any] | None = None
        self._cached_living_loop_status: dict[str, Any] | None = None
        self._cached_policy_actuator_status: dict[str, Any] | None = None
        self._cached_subcortical_self_repair_surface: dict[str, Any] | None = None
        self._cached_subcortical_self_repair_evaluation_surface: dict[str, Any] | None = None
        self._cached_subcortical_structural_plasticity_surface: dict[str, Any] | None = None

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
            "tick_tokens": int(
                terminus_runtime.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)
                or DEFAULT_BRAIN_TICK_TOKENS
            ),
            "sleep_interval_seconds": float(
                terminus_runtime.get("sleep_interval_seconds", 0.0) or 0.0
            ),
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
        retired_runtime_path_source = (
            terminus_runtime.get("retired_runtime_path")
            or terminus_runtime.get("cortex")
            if isinstance(terminus_runtime, Mapping)
            else {}
        )
        retired_runtime_path_available = bool(
            isinstance(retired_runtime_path_source, Mapping) and retired_runtime_path_source.get("enabled")
        )
        retired_runtime_path_retired = bool(
            isinstance(retired_runtime_path_source, Mapping) and retired_runtime_path_source.get("retired")
        )
        retired_runtime_path = {
            "name": "cortex",
            "available": retired_runtime_path_available,
            "retired": retired_runtime_path_retired,
            "active_runtime_requirement": False,
            "operator_surface": False,
            "compatibility_aliases": ["cortex_available", "cortex_retired"],
        }
        retired_runtime_path_evidence = {
            "name": "cortex",
            "enabled": retired_runtime_path_available,
            "retired": retired_runtime_path_retired,
            "active_runtime_requirement": False,
            "operator_surface": False,
            "compatibility_aliases": ["cortex_enabled", "cortex_retired"],
        }
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
        if fill_fraction >= 0.85:
            pressure = "high"
        elif fill_fraction >= 0.50:
            pressure = "medium"
        else:
            pressure = "low"

        working_set_policy = {
            "high_threshold": 0.85,
            "target_fill": 0.70,
            "capacity_increase_recommended": False,
            "replay_fact_promotion_allowed": False,
            "decision": self._working_set_decision(pressure),
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
        subcortex_spike_health = self._trainer.model.competitive.spike_health_report()
        self_repair_surface = build_subcortical_self_repair_surface(subcortex_spike_health)
        self_repair_gate = (
            self_repair_surface.get("promotion_gate")
            if isinstance(self_repair_surface.get("promotion_gate"), Mapping)
            else {}
        )
        self_repair_gate = {
            "surface": self_repair_surface.get("surface"),
            "artifact_kind": self_repair_surface.get("artifact_kind"),
            "source": self_repair_surface.get("source"),
            "advisory": bool(self_repair_surface.get("advisory")),
            "executable": bool(self_repair_surface.get("executable")),
            "promotion_status": self_repair_gate.get("status"),
            "next_gate": self_repair_gate.get("next_gate"),
            "eligible_for_action": bool(self_repair_gate.get("eligible_for_action")),
            "eligible_for_fact_promotion": bool(self_repair_gate.get("eligible_for_fact_promotion")),
            "eligible_for_replay_review": bool(self_repair_gate.get("eligible_for_replay_review")),
            "eligible_for_structural_mutation": bool(self_repair_gate.get("eligible_for_structural_mutation")),
            "candidate_count": int(
                (self_repair_surface.get("promotion_summary") or {}).get("candidate_count", 0)
                if isinstance(self_repair_surface.get("promotion_summary"), Mapping)
                else 0
            ),
        }
        structural_surface = build_subcortical_structural_plasticity_surface(
            self._concept_store_snapshot_fn(),
            self._runtime_scope_report_locked(),
        )
        structural_gate = (
            structural_surface.get("promotion_gate")
            if isinstance(structural_surface.get("promotion_gate"), Mapping)
            else {}
        )
        structural_concept_growth = (
            structural_surface.get("concept_growth")
            if isinstance(structural_surface.get("concept_growth"), Mapping)
            else {}
        )
        structural_device_evidence = (
            structural_surface.get("device_evidence")
            if isinstance(structural_surface.get("device_evidence"), Mapping)
            else {}
        )
        structural_local_plasticity = (
            structural_surface.get("local_plasticity")
            if isinstance(structural_surface.get("local_plasticity"), Mapping)
            else {}
        )
        structural_plasticity_gate = {
            "surface": structural_surface.get("surface"),
            "artifact_kind": structural_surface.get("artifact_kind"),
            "source": structural_surface.get("source"),
            "advisory": bool(structural_surface.get("advisory")),
            "executable": bool(structural_surface.get("executable")),
            "mutates_runtime_state": bool(structural_surface.get("mutates_runtime_state")),
            "promotion_status": structural_gate.get("status"),
            "next_gate": structural_gate.get("next_gate"),
            "eligible_for_action": bool(structural_gate.get("eligible_for_action")),
            "eligible_for_fact_promotion": bool(structural_gate.get("eligible_for_fact_promotion")),
            "eligible_for_structural_mutation": bool(structural_gate.get("eligible_for_structural_mutation")),
            "ready_case_count": int(structural_gate.get("ready_case_count", 0) or 0),
            "case_count": int(structural_gate.get("case_count", 0) or 0),
            "concept_growth_ready": bool(structural_concept_growth.get("growth_ready")),
            "binding_report_available": bool(structural_device_evidence.get("binding_report_available")),
            "local_plasticity_report_available": bool(
                structural_device_evidence.get("local_plasticity_report_available")
            ),
            "local_plasticity_homeostatic_state_available": bool(
                structural_local_plasticity.get("homeostatic_state_available")
            ),
        }
        cognitive_signal = (
            self._cognitive_signal_state_fn()
            if self._cognitive_signal_state_fn is not None
            else {}
        )
        snn_language_surface = build_snn_language_readiness_surface(
            cognitive_signal,
            self._runtime_scope_report_locked(),
        )
        snn_language_gate = (
            snn_language_surface.get("promotion_gate")
            if isinstance(snn_language_surface.get("promotion_gate"), Mapping)
            else {}
        )
        readiness_checks = (
            snn_language_surface.get("readiness_checks")
            if isinstance(snn_language_surface.get("readiness_checks"), Mapping)
            else {}
        )
        snn_language_readiness_gate = {
            "surface": snn_language_surface.get("surface"),
            "artifact_kind": snn_language_surface.get("artifact_kind"),
            "source": snn_language_surface.get("source"),
            "advisory": bool(snn_language_surface.get("advisory")),
            "executable": bool(snn_language_surface.get("executable")),
            "mutates_runtime_state": bool(snn_language_surface.get("mutates_runtime_state")),
            "not_cognition_substrate": bool(snn_language_surface.get("not_cognition_substrate")),
            "retired_runtime_dependency": bool(snn_language_surface.get("retired_runtime_dependency")),
            "promotion_status": snn_language_gate.get("status"),
            "next_gate": snn_language_gate.get("next_gate"),
            "eligible_for_action": bool(snn_language_gate.get("eligible_for_action")),
            "eligible_for_fact_promotion": bool(snn_language_gate.get("eligible_for_fact_promotion")),
            "eligible_for_cognition_substrate": bool(snn_language_gate.get("eligible_for_cognition_substrate")),
            "eligible_for_language_generation": bool(snn_language_gate.get("eligible_for_language_generation")),
            "requires_hecsn_owned_implementation": bool(
                (snn_language_surface.get("safety_invariants") or {}).get("requires_hecsn_owned_implementation")
                if isinstance(snn_language_surface.get("safety_invariants"), Mapping)
                else False
            ),
            "grounded_language_surface_available": bool(readiness_checks.get("grounded_language_surface_available")),
            "local_snn_language_generator_available": bool(
                readiness_checks.get("local_snn_language_generator_available")
            ),
            "activation_sparsity_report_available": bool(
                readiness_checks.get("activation_sparsity_report_available")
            ),
            "grounding_support_report_available": bool(
                readiness_checks.get("grounding_support_report_available")
            ),
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "recommended_action": recommended_action,
            "retired_runtime_path": retired_runtime_path,
            "retired_runtime_path_available": retired_runtime_path_available,
            "retired_runtime_path_retired": retired_runtime_path_retired,
            "cortex_available": retired_runtime_path_available,
            "cortex_retired": retired_runtime_path_retired,
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
                "retired_runtime_path": retired_runtime_path_evidence,
                "retired_runtime_path_enabled": retired_runtime_path_available,
                "retired_runtime_path_retired": retired_runtime_path_retired,
                "cortex_enabled": retired_runtime_path_available,
                "cortex_retired": retired_runtime_path_retired,
                "replay_endpoint": replay_endpoint,
                "source_configuration_hash": source_configuration["configuration_hash"],
                "subcortex_spike_health": subcortex_spike_health,
                "self_repair_gate": self_repair_gate,
                "structural_plasticity_gate": structural_plasticity_gate,
                "snn_language_readiness_gate": snn_language_readiness_gate,
            },
        }

    @staticmethod
    def _working_set_decision(pressure: str) -> str:
        if pressure == "high":
            return "throttle_ingestion_and_prioritize_consolidation"
        if pressure == "medium":
            return "watch_working_set_growth"
        return "continue_monitoring"

    def _state_norm(self) -> float:
        return float(torch.linalg.norm(self._trainer.context_state().float()).item())

    def _last_trace_fields(self) -> tuple[int, str | None, str | None]:
        last_trace = self._trace_history[0] if self._trace_history else None
        return (
            int(len(self._trace_history)),
            None if last_trace is None else str(last_trace.get("trace_id")),
            None if last_trace is None else str(last_trace.get("created_at")),
        )

    def _runtime_mutation_payload(self) -> dict[str, Any]:
        return {
            **self._runtime_state.mutation_summary(),
            "token_count": int(self._trainer.token_count),
        }

    def _runtime_scope_report_locked(self) -> dict[str, Any]:
        """Return model runtime scope enriched with trainer-owned encoder evidence."""
        runtime_scope = deepcopy(self._trainer.model.runtime_scope_report())
        encoder = getattr(self._trainer, "encoder", None)
        encoder_report = encoder.device_report() if hasattr(encoder, "device_report") else None
        cuda_runtime = runtime_scope.get("cuda_first_runtime")
        if isinstance(cuda_runtime, dict):
            cuda_runtime["encoder_device_report"] = deepcopy(encoder_report)
        else:
            runtime_scope["cuda_first_runtime"] = {"encoder_device_report": deepcopy(encoder_report)}
        return runtime_scope

    # ------------------------------------------------------------------
    # Internal snapshot builders (must be called under self._lock)
    # ------------------------------------------------------------------

    def _status_snapshot_locked(self) -> dict[str, Any]:
        """Build the full status snapshot. Caller MUST hold self._lock."""
        terminus_runtime = self._brain_runtime_snapshot_fn()
        runtime_mutation = self._runtime_state.mutation_summary()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size, last_trace_id, last_trace_created_at = self._last_trace_fields()

        return {
            "checkpoint_path": str(self._checkpoint_path_str),
            **runtime_mutation,
            "token_count": int(self._trainer.token_count),
            "last_winner": (
                None if self._trainer.last_winner is None else int(self._trainer.last_winner)
            ),
            "context_supported": bool(self._trainer.model.context_layer is not None),
            "context_state_norm": self._state_norm(),
            "trace_history_size": trace_history_size,
            "trace_storage_dir": str(self._trace_dir_str),
            "last_trace_id": last_trace_id,
            "last_trace_created_at": last_trace_created_at,
            "checkpoint_metadata": deepcopy(self._metadata),
            "dopamine": float(self._trainer.model.surprise.dopamine),
            "serotonin": float(self._trainer.model.surprise.serotonin),
            "acetylcholine": float(self._trainer.model.surprise.acetylcholine),
            "norepinephrine": float(self._trainer.model.surprise.norepinephrine),
            "runtime_scope": self._runtime_scope_report_locked(),
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
            "runtime_scope": self._runtime_scope_report_locked(),
            "memory_store": memory_store,
            "replay_dataset_summary": replay_dataset_summary,
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
            ),
        }

    def _telemetry_snapshot_locked(self) -> dict[str, Any]:
        """Build the telemetry dict. Caller MUST hold self._lock."""
        runtime_mutation = self._runtime_state.mutation_summary()
        current_rev = int(runtime_mutation["state_revision"])
        cortex_active = self._cortex_active_fn() if self._cortex_active_fn is not None else False
        # Revision-keyed cache reuse: when the cortex is inactive and the
        # revision hasn't changed, the cached telemetry is still valid.
        if not cortex_active and self._cached_telemetry is not None and self._cached_telemetry_rev == current_rev:
            return self._cached_telemetry

        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size, last_trace_id, last_trace_created_at = self._last_trace_fields()
        drift_value = (
            self._trainer._cached_drift
            if self._trainer._cached_drift is not None
            else self._trainer.model.memory_store.compute_drift(
                self._trainer.last_winner
                if self._trainer.config.use_winner_local_drift
                else None
            )
        )
        terminus_runtime = self._brain_runtime_snapshot_fn()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)

        animation = self._animation_snapshot_fn() if self._animation_snapshot_fn is not None else {}

        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(self._checkpoint_path_str),
            "dirty_state": bool(runtime_mutation["dirty_state"]),
            "state_revision": current_rev,
            "token_count": int(self._trainer.token_count),
            "last_winner": None if self._trainer.last_winner is None else int(self._trainer.last_winner),
            "context_state_norm": self._state_norm(),
            "trace_history_size": trace_history_size,
            "last_trace_id": last_trace_id,
            "last_trace_created_at": last_trace_created_at,
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
            "drift_floor": float(
                self._trainer.current_rolling_drift_floor
                if self._trainer.current_rolling_drift_floor is not None
                else drift_value
            ),
            "grounding_confidence": {
                w: round(c, 4) for w, c in self._trainer.word_grounding_confidence.items()
            },
            "n_visual_signatures": len(self._trainer.word_visual_signature),
            "n_audio_signatures": len(self._trainer.word_audio_signature),
            "cross_modal_visual_confidence": (
                float(self._trainer.model.cross_modal.visual_confidence.mean().item())
                if self._trainer.model.cross_modal is not None
                else None
            ),
            "cross_modal_audio_confidence": (
                float(self._trainer.model.cross_modal.audio_confidence.mean().item())
                if self._trainer.model.cross_modal is not None
                else None
            ),
            "animation": animation,
            "terminus_runtime": terminus_runtime,
            "runtime_scope": self._runtime_scope_report_locked(),
            "memory_store": memory_store,
            "replay_dataset_summary": replay_dataset_summary,
        }
        self._cached_telemetry = snapshot
        self._cached_telemetry_rev = current_rev
        return snapshot

    # ------------------------------------------------------------------
    # Public surfaces
    # ------------------------------------------------------------------

    def _read_snapshot(
        self,
        *,
        fresh_wait_seconds: float | None,
        cached_snapshot: dict[str, Any] | None,
        snapshot_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a snapshot with non-blocking cached fallback semantics."""
        if fresh_wait_seconds is None:
            acquired = self._lock.acquire(timeout=DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS)
            if not acquired:
                if cached_snapshot is not None:
                    return cached_snapshot
                self._lock.acquire()
        else:
            deadline = time.perf_counter() + max(0.0, float(fresh_wait_seconds))
            acquired = False
            while time.perf_counter() < deadline:
                remaining = max(0.0, deadline - time.perf_counter())
                if self._lock.acquire(timeout=min(DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS, remaining)):
                    acquired = True
                    break
            if not acquired:
                self._lock.acquire()
        try:
            return snapshot_fn()
        finally:
            self._lock.release()

    def status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Return the status snapshot (non-blocking by default).

        When ``fresh_wait_seconds`` is ``None`` the call tries to acquire the
        lock for up to 150 ms. If that fails and a cached snapshot exists, the
        cached copy is returned. When a caller explicitly requests a fresh
        snapshot, the method retries until the deadline and then blocks rather
        than silently serving stale data.
        """
        result = self._read_snapshot(
            fresh_wait_seconds=fresh_wait_seconds,
            cached_snapshot=self._cached_status,
            snapshot_fn=self._status_snapshot_locked,
        )
        self._cached_status = result
        return result

    def terminus_status(self, *, fresh_wait_seconds: float | None = None) -> dict[str, Any]:
        """Return the terminus status snapshot (non-blocking by default).

        Polling semantics mirror ``status()``: non-blocking default with cached
        fallback, and blocking retry when an explicit freshness deadline is
        requested.
        """
        result = self._read_snapshot(
            fresh_wait_seconds=fresh_wait_seconds,
            cached_snapshot=self._cached_terminus_status,
            snapshot_fn=self._terminus_status_snapshot_locked,
        )
        self._cached_terminus_status = result
        return result

    # ------------------------------------------------------------------
    # Sensory previews
    # ------------------------------------------------------------------

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
        """Return recent sensory preview payloads (read-only projection).

        Parameters
        ----------
        limit:
            Maximum number of preview items to return.  Defaults to 6.

        """
        return self._read_sensory_previews(limit=limit)

    def _read_sensory_previews(self, limit: int) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS)
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
                        "visual": self._sensory_media_payload(item.get("visual")),
                        "audio": self._sensory_media_payload(item.get("audio")),
                    }
                )
            return {
                "count": int(len(self._sensory_preview_history)),
                "latest_preview_id": (
                    None
                    if not self._sensory_preview_history
                    else str(self._sensory_preview_history[0].get("preview_id", ""))
                ),
                "previews": previews,
            }
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Architecture summary
    # ------------------------------------------------------------------

    def architecture_summary(self) -> dict[str, Any]:
        """Return the current architecture summary (read-only projection).

        Delegates to the injected ``architecture_snapshot_fn`` callback so the
        read model stays decoupled from the full manager surface.

        """
        return self._architecture_snapshot_fn()

    def telemetry_snapshot(self) -> dict[str, Any]:
        """Return the telemetry snapshot (non-blocking with lock-contention fallback).

        Non-blocking: return cached data when the brain loop holds the lock
        (prevents SSE/API starvation during training or HF network I/O). When
        the cortex is inactive and the state revision hasn't changed, the
        previously cached telemetry snapshot is reused instead of rebuilding.
        """
        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_telemetry,
            snapshot_fn=self._telemetry_snapshot_locked,
        )

    # ------------------------------------------------------------------
    # Living Loop and Policy Actuator snapshots
    # ------------------------------------------------------------------

    def living_loop_status(self) -> dict[str, Any]:
        """Return the living loop status snapshot (non-blocking with lock-contention fallback).

        Delegates to the injected ``living_loop_status_fn`` callback so the
        read model stays decoupled from the full manager surface. The callback
        is called under the shared lock; when the lock is contended the
        previously cached result is returned.
        """
        if self._living_loop_status_fn is None:
            return {
                "living_loop": {},
                **self._runtime_mutation_payload(),
            }
        living_loop_status_fn = self._living_loop_status_fn

        def _build_locked() -> dict[str, Any]:
            payload = living_loop_status_fn()
            payload = self._attach_living_loop_control_candidates(payload)
            payload = self._attach_living_loop_self_repair_candidates(payload)
            return {
                **payload,
                **self._runtime_mutation_payload(),
            }

        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_living_loop_status,
            snapshot_fn=_build_locked,
        )
        self._cached_living_loop_status = result
        return result

    def _attach_living_loop_control_candidates(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Attach read-only Subcortex control candidates to the living-loop sidecar."""
        enriched = dict(payload)
        living_loop = dict(enriched.get("living_loop") or {})
        cognitive_signal = self.cognitive_signal_state()
        control_candidates = cognitive_signal.get("subcortical_deliberation")
        if isinstance(control_candidates, Mapping):
            living_loop["subcortical_control_candidates"] = dict(control_candidates)
            enriched["living_loop"] = living_loop
        return enriched

    def _subcortical_self_repair_surface(self) -> dict[str, Any]:
        """Build advisory self-repair candidates from current spike-health evidence."""
        spike_health = self._trainer.model.competitive.spike_health_report()
        return build_subcortical_self_repair_surface(spike_health)

    def _subcortical_self_repair_evaluation_surface(self) -> dict[str, Any]:
        """Build a read-only self-repair evaluation artifact from spike-health evidence."""
        spike_health = self._trainer.model.competitive.spike_health_report()
        return build_subcortical_self_repair_evaluation_surface(spike_health)

    def subcortical_self_repair_surface(self) -> dict[str, Any]:
        """Return the reviewable self-repair gate artifact without mutating runtime state."""
        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_subcortical_self_repair_surface,
            snapshot_fn=self._subcortical_self_repair_surface,
        )
        self._cached_subcortical_self_repair_surface = result
        return result

    def subcortical_self_repair_evaluation_surface(self) -> dict[str, Any]:
        """Return the read-only self-repair evaluation artifact without mutating runtime state."""
        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_subcortical_self_repair_evaluation_surface,
            snapshot_fn=self._subcortical_self_repair_evaluation_surface,
        )
        self._cached_subcortical_self_repair_evaluation_surface = result
        return result

    def _subcortical_structural_plasticity_surface(self) -> dict[str, Any]:
        """Build read-only structural-plasticity promotion evidence."""
        return build_subcortical_structural_plasticity_surface(
            self._concept_store_snapshot_fn(),
            self._runtime_scope_report_locked(),
        )

    def subcortical_structural_plasticity_surface(self) -> dict[str, Any]:
        """Return structural-plasticity review evidence without mutating runtime state."""
        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_subcortical_structural_plasticity_surface,
            snapshot_fn=self._subcortical_structural_plasticity_surface,
        )
        self._cached_subcortical_structural_plasticity_surface = result
        return result

    def _attach_living_loop_self_repair_candidates(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Attach read-only Subcortex self-repair candidates to the living-loop sidecar."""
        enriched = dict(payload)
        living_loop = dict(enriched.get("living_loop") or {})
        living_loop["subcortical_self_repair_candidates"] = self._subcortical_self_repair_surface()
        enriched["living_loop"] = living_loop
        return enriched

    def policy_actuator_status(self) -> dict[str, Any]:
        """Return the policy actuator status snapshot (non-blocking with lock-contention fallback).

        Delegates to the injected ``policy_actuator_status_fn`` callback so
        the read model stays decoupled from the full manager surface. The
        callback is called under the shared lock; when the lock is contended
        the previously cached result is returned.
        """
        if self._policy_actuator_status_fn is None:
            return {
                "schema_version": 1,
                "recommendation": "no_policy_actuator_configured",
                "action": "none",
                "reasons": [],
                "advisory": True,
                "executable": False,
            }
        policy_actuator_status_fn = self._policy_actuator_status_fn

        def _build_locked() -> dict[str, Any]:
            payload = self._attach_policy_control_candidates(policy_actuator_status_fn())
            return self._attach_policy_self_repair_candidates(payload)

        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_policy_actuator_status,
            snapshot_fn=_build_locked,
        )
        self._cached_policy_actuator_status = result
        return result

    def _attach_policy_control_candidates(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Attach advisory Subcortex control candidates to policy status."""
        enriched = dict(payload)
        cognitive_signal = self.cognitive_signal_state()
        control_candidates = cognitive_signal.get("subcortical_deliberation")
        if isinstance(control_candidates, Mapping):
            enriched["subcortical_control_candidates"] = {
                **dict(control_candidates),
                "advisory": True,
                "executable": False,
            }
        return enriched

    def _attach_policy_self_repair_candidates(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Attach advisory Subcortex self-repair candidates to policy status."""
        enriched = dict(payload)
        enriched["subcortical_self_repair_candidates"] = {
            **self._subcortical_self_repair_surface(),
            "advisory": True,
            "executable": False,
        }
        return enriched

    # ------------------------------------------------------------------
    # Cognitive Signal state (cached with short-timeout fallback)
    # ------------------------------------------------------------------

    def cognitive_signal_state(self) -> dict[str, Any]:
        """Expose recent Subcortex predictive/surprise signals.

        Uses a short lock timeout (50 ms) so signal readers do not block
        runtime progress. When the lock is contended, the cached result is
        returned. On first call with no cache, blocks until the lock is available.

        The injected ``cognitive_signal_state_fn`` callback is called under
        lock to build the fresh payload, and the result is cached for
        lock-contention fallback.
        """
        if self._cognitive_signal_state_fn is None:
            return {}
        cognitive_signal_state_fn = self._cognitive_signal_state_fn

        acquired = self._lock.acquire(timeout=DEFAULT_COGNITIVE_SIGNAL_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            if self._cached_cognitive_signal_state is not None:
                return self._cached_cognitive_signal_state
            self._lock.acquire()
        try:
            payload = attach_cognitive_signal_language_surface(cognitive_signal_state_fn())
            self._cached_cognitive_signal_state = payload
            self._cached_cortex_signal_state = payload
            return payload
        finally:
            self._lock.release()

    def cortex_signal_state(self) -> dict[str, Any]:
        """Compatibility wrapper for the retired Cortex signal name."""
        return self.cognitive_signal_state()

    def subcortical_language_surface(self) -> dict[str, Any]:
        """Return the Cognitive Signal language surface without changing runtime state."""
        return dict(self.cognitive_signal_state().get("subcortical_language") or {})

    def subcortical_deliberation_surface(self) -> dict[str, Any]:
        """Return bounded Cognitive Signal deliberation candidates."""
        return dict(self.cognitive_signal_state().get("subcortical_deliberation") or {})

    def _snn_language_readiness_surface(self) -> dict[str, Any]:
        """Build a read-only SNN-native language readiness artifact."""
        return build_snn_language_readiness_surface(
            self.cognitive_signal_state(),
            self._runtime_scope_report_locked(),
        )

    def snn_language_readiness_surface(self) -> dict[str, Any]:
        """Return readiness evidence for future SNN-native language generation."""
        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_snn_language_readiness_surface,
            snapshot_fn=self._snn_language_readiness_surface,
        )
        self._cached_snn_language_readiness_surface = result
        return result
