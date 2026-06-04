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
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence
import base64
import hashlib
import json
import time

import torch

_SNN_LANGUAGE_NEURON_COUNT = 64
_SNN_LANGUAGE_SPARSE_EDGE_BUDGET = 256
_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET = 16
_SNN_LANGUAGE_CAPACITY_SURFACE = "snn_language_capacity_state.v1"

from hecsn.semantics import (
    attach_cognitive_signal_language_surface,
    build_snn_language_readiness_surface,
    build_snn_language_evaluation_surface,
    build_snn_language_training_readiness_surface,
    build_subcortical_self_repair_evaluation_surface,
    build_subcortical_self_repair_surface,
    build_subcortical_structural_mutation_design,
    build_subcortical_structural_mutation_preflight,
    build_subcortical_structural_plasticity_surface,
    build_spike_language_plasticity_application_design,
    build_spike_language_plasticity_pressure,
    build_spike_language_plasticity_shadow_delta,
    build_snn_language_readout_emission,
    build_snn_language_transition_memory_sleep_policy,
    build_snn_language_transition_memory_prediction_evaluation,
    build_snn_language_transition_memory_regeneration_proposal,
    evaluate_spike_language_adapter_heldout,
    evaluate_spike_language_plasticity_live_application_preflight,
    evaluate_spike_language_plasticity_live_application_readiness,
    evaluate_spike_language_plasticity_shadow_application,
    evaluate_spike_language_plasticity_replay,
    evaluate_spike_language_sequence_mismatch,
    evaluate_spike_language_trainer_dry_run,
    evaluate_snn_language_readout_rollout_replay,
    evaluate_subcortical_structural_plasticity_isolated,
    generate_snn_language_readout_draft,
    predict_spike_language_sequence,
    rollout_snn_language_readout_candidate,
    run_spike_language_plasticity_replay_experiment,
    run_spike_language_trainer_dry_run,
    run_spike_language_plasticity_trial,
)
from hecsn.service.runtime_state import RuntimeState

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_LOCK_ACQUIRE_TIMEOUT_SECONDS = 0.15
DEFAULT_COGNITIVE_SIGNAL_LOCK_TIMEOUT_SECONDS = 0.05


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
    sleep_plasticity_autonomy_proposal_fn: Callable that returns the current
        non-executing sleep-plasticity autonomy proposal. Called under lock and
        attached only as advisory sidecar evidence.
    sleep_plasticity_scheduler_installation_autonomy_proposal_fn: Callable that
        returns the current non-executing scheduler-installation autonomy
        proposal. Called under lock and attached only as advisory sidecar
        evidence.
    due_cycle_bounded_replay_selection_proposal_fn: Callable that returns the
        current non-executing due-cycle bounded replay-selection proposal.
        Called under lock and attached only as advisory sidecar evidence.
    language_plasticity_state_fn: Callable that returns the current SNN language
        plasticity state. Called under lock and summarized only as non-executing
        server-state binding evidence for readout rollout.
    readout_ledger_state_fn: Callable that returns the current SNN language
        readout ledger state. Called under lock and summarized only as
        non-executing rollout rehearsal/consolidation path evidence.
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
        animation_snapshot_fn: Callable[[], dict[str, Any]] | None = None,
        living_loop_status_fn: Callable[[], dict[str, Any]] | None = None,
        policy_actuator_status_fn: Callable[[], dict[str, Any]] | None = None,
        cognitive_signal_state_fn: Callable[[], dict[str, Any]] | None = None,
        sleep_plasticity_autonomy_proposal_fn: Callable[[], dict[str, Any]] | None = None,
        sleep_plasticity_scheduler_installation_autonomy_proposal_fn: Callable[[], dict[str, Any]] | None = None,
        due_cycle_bounded_replay_selection_proposal_fn: Callable[[], dict[str, Any]] | None = None,
        language_plasticity_state_fn: Callable[[], dict[str, Any]] | None = None,
        readout_ledger_state_fn: Callable[[], dict[str, Any]] | None = None,
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
        self._animation_snapshot_fn = animation_snapshot_fn
        self._living_loop_status_fn = living_loop_status_fn
        self._policy_actuator_status_fn = policy_actuator_status_fn
        self._cognitive_signal_state_fn = cognitive_signal_state_fn
        self._sleep_plasticity_autonomy_proposal_fn = sleep_plasticity_autonomy_proposal_fn
        self._sleep_plasticity_scheduler_installation_autonomy_proposal_fn = (
            sleep_plasticity_scheduler_installation_autonomy_proposal_fn
        )
        self._due_cycle_bounded_replay_selection_proposal_fn = (
            due_cycle_bounded_replay_selection_proposal_fn
        )
        self._language_plasticity_state_fn = language_plasticity_state_fn
        self._readout_ledger_state_fn = readout_ledger_state_fn

        # Cache state — owned by the read model
        self._cached_status: dict[str, Any] | None = None
        self._cached_terminus_status: dict[str, Any] | None = None
        self._cached_telemetry: dict[str, Any] | None = None
        self._cached_telemetry_rev: int = -1
        self._cached_cognitive_signal_state: dict[str, Any] | None = None
        self._cached_snn_language_readiness_surface: dict[str, Any] | None = None
        self._cached_snn_language_evaluation_surface: dict[str, Any] | None = None
        self._cached_living_loop_status: dict[str, Any] | None = None
        self._cached_policy_actuator_status: dict[str, Any] | None = None
        self._cached_subcortical_self_repair_surface: dict[str, Any] | None = None
        self._cached_subcortical_self_repair_evaluation_surface: dict[str, Any] | None = None
        self._cached_subcortical_structural_plasticity_surface: dict[str, Any] | None = None

    def rebind_runtime(
        self,
        *,
        trainer: Any,
        metadata: Mapping[str, Any],
        checkpoint_path_str: str,
    ) -> None:
        self._trainer = trainer
        self._metadata = dict(metadata)
        self._checkpoint_path_str = checkpoint_path_str
        self._cached_status = None
        self._cached_terminus_status = None
        self._cached_telemetry = None
        self._cached_telemetry_rev = -1
        self._cached_cognitive_signal_state = None
        self._cached_snn_language_readiness_surface = None
        self._cached_snn_language_evaluation_surface = None
        self._cached_living_loop_status = None
        self._cached_policy_actuator_status = None
        self._cached_subcortical_self_repair_surface = None
        self._cached_subcortical_self_repair_evaluation_surface = None
        self._cached_subcortical_structural_plasticity_surface = None

    # ------------------------------------------------------------------
    # Static helpers reused from RuntimeStatusCore
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
        sleep_plasticity_autonomy_proposal: Mapping[str, Any] | None = None,
        sleep_plasticity_scheduler_installation_autonomy_proposal: Mapping[str, Any] | None = None,
        due_cycle_bounded_replay_selection_proposal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Runtime Truth contract. Reads self._trainer for token_count only."""
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
        self_repair_evaluation_surface = build_subcortical_self_repair_evaluation_surface(
            subcortex_spike_health
        )
        self_repair_evaluation_gate_surface = (
            self_repair_evaluation_surface.get("evaluation_gate")
            if isinstance(self_repair_evaluation_surface.get("evaluation_gate"), Mapping)
            else {}
        )
        self_repair_evaluation_safety = (
            self_repair_evaluation_surface.get("safety_invariants")
            if isinstance(self_repair_evaluation_surface.get("safety_invariants"), Mapping)
            else {}
        )
        self_repair_evaluation_gate = {
            "surface": self_repair_evaluation_surface.get("surface"),
            "artifact_kind": self_repair_evaluation_surface.get("artifact_kind"),
            "source": self_repair_evaluation_surface.get("source"),
            "advisory": bool(self_repair_evaluation_surface.get("advisory")),
            "executable": bool(self_repair_evaluation_surface.get("executable")),
            "mutates_runtime_state": bool(
                self_repair_evaluation_surface.get("mutates_runtime_state")
            ),
            "promotion_status": self_repair_evaluation_gate_surface.get("status"),
            "next_gate": self_repair_evaluation_gate_surface.get("next_gate"),
            "ready_case_count": int(
                self_repair_evaluation_gate_surface.get("ready_case_count", 0) or 0
            ),
            "case_count": int(
                self_repair_evaluation_gate_surface.get("case_count", 0) or 0
            ),
            "eligible_for_action": bool(
                self_repair_evaluation_gate_surface.get("eligible_for_action")
            ),
            "eligible_for_fact_promotion": bool(
                self_repair_evaluation_gate_surface.get("eligible_for_fact_promotion")
            ),
            "eligible_for_replay_review": bool(
                self_repair_evaluation_gate_surface.get("eligible_for_replay_review")
            ),
            "eligible_for_structural_mutation": bool(
                self_repair_evaluation_gate_surface.get("eligible_for_structural_mutation")
            ),
            "requires_operator_approval": bool(
                self_repair_evaluation_gate_surface.get("requires_operator_approval")
                or self_repair_evaluation_safety.get("requires_operator_approval")
            ),
            "requires_isolated_replay_or_deep_sleep": bool(
                self_repair_evaluation_safety.get("requires_isolated_replay_or_deep_sleep")
            ),
            "requires_runtime_truth_improvement": bool(
                self_repair_evaluation_safety.get("requires_runtime_truth_improvement")
            ),
            "requires_device_evidence": bool(
                self_repair_evaluation_safety.get("requires_device_evidence")
            ),
            "success_evidence": list(
                str(item) for item in list(
                    self_repair_evaluation_surface.get("success_evidence") or []
                )[:8]
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
        structural_binding_devices = (
            structural_device_evidence.get("binding_devices")
            if isinstance(structural_device_evidence.get("binding_devices"), Mapping)
            else {}
        )
        structural_local_devices = (
            structural_device_evidence.get("local_plasticity_devices")
            if isinstance(structural_device_evidence.get("local_plasticity_devices"), Mapping)
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
            "eligible_for_replay_review": bool(structural_gate.get("eligible_for_replay_review")),
            "eligible_for_structural_mutation": bool(structural_gate.get("eligible_for_structural_mutation")),
            "requires_operator_approval": bool(structural_gate.get("requires_operator_approval")),
            "requires_isolated_evaluation": bool(
                (structural_surface.get("safety_invariants") or {}).get("requires_isolated_evaluation")
                if isinstance(structural_surface.get("safety_invariants"), Mapping)
                else False
            ),
            "requires_runtime_truth_improvement": bool(
                (structural_surface.get("safety_invariants") or {}).get("requires_runtime_truth_improvement")
                if isinstance(structural_surface.get("safety_invariants"), Mapping)
                else False
            ),
            "requires_reversible_mutation_ledger": bool(
                (structural_surface.get("safety_invariants") or {}).get("requires_reversible_mutation_ledger")
                if isinstance(structural_surface.get("safety_invariants"), Mapping)
                else False
            ),
            "requires_device_evidence": bool(
                (structural_surface.get("safety_invariants") or {}).get("requires_device_evidence")
                if isinstance(structural_surface.get("safety_invariants"), Mapping)
                else False
            ),
            "success_evidence": list(
                str(item) for item in list(structural_surface.get("success_evidence") or [])[:8]
            ),
            "ready_case_count": int(structural_gate.get("ready_case_count", 0) or 0),
            "case_count": int(structural_gate.get("case_count", 0) or 0),
            "concept_growth_ready": bool(structural_concept_growth.get("growth_ready")),
            "binding_report_available": bool(structural_device_evidence.get("binding_report_available")),
            "binding_device_keys": sorted(str(key) for key in structural_binding_devices.keys()),
            "local_plasticity_report_available": bool(
                structural_device_evidence.get("local_plasticity_report_available")
            ),
            "local_plasticity_device_keys": sorted(str(key) for key in structural_local_devices.keys()),
            "observed_structural_device_key_count": len(structural_binding_devices) + len(structural_local_devices),
            "local_plasticity_eligibility_traces_available": bool(
                structural_local_plasticity.get("eligibility_traces_available")
            ),
            "local_plasticity_homeostatic_state_available": bool(
                structural_local_plasticity.get("homeostatic_state_available")
            ),
            "local_plasticity_spike_backend": structural_local_plasticity.get("spike_backend"),
            "local_plasticity_rule": structural_local_plasticity.get("plasticity_rule"),
            "local_plasticity_spike_health_risk": bool(
                structural_local_plasticity.get("spike_health_risk")
            ),
            "local_plasticity_synaptic_validation_available": bool(
                structural_local_plasticity.get("synaptic_validation_available")
            ),
            "local_plasticity_synaptic_validation_passed": bool(
                structural_local_plasticity.get("synaptic_validation_passed")
            ),
            "local_plasticity_synaptic_validation_failed": bool(
                structural_local_plasticity.get("synaptic_validation_failed")
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
            "hecsn_spike_readout_evidence_available": bool(
                readiness_checks.get("hecsn_spike_readout_evidence_available")
            ),
            "hecsn_spike_readout_grounded": bool(readiness_checks.get("hecsn_spike_readout_grounded")),
            "hecsn_spike_readout_non_generative": bool(
                readiness_checks.get("hecsn_spike_readout_non_generative")
            ),
            "hecsn_spike_readout_device_evidence_available": bool(
                readiness_checks.get("hecsn_spike_readout_device_evidence_available")
            ),
            "hecsn_spike_decoder_probe_available": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_available")
            ),
            "hecsn_spike_decoder_probe_owned": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_owned")
            ),
            "hecsn_spike_decoder_probe_non_generative": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_non_generative")
            ),
            "hecsn_spike_decoder_probe_sparse": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_sparse")
            ),
            "hecsn_spike_decoder_probe_device_evidence_available": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_device_evidence_available")
            ),
            "hecsn_spike_decoder_probe_grounding_supported": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_grounding_supported")
            ),
            "hecsn_spike_decoder_probe_temporal_state": bool(
                readiness_checks.get("hecsn_spike_decoder_probe_temporal_state")
            ),
            "hecsn_spike_language_neuron_adapter_available": bool(
                readiness_checks.get("hecsn_spike_language_neuron_adapter_available")
            ),
            "hecsn_spike_language_neuron_adapter_owned": bool(
                readiness_checks.get("hecsn_spike_language_neuron_adapter_owned")
            ),
            "hecsn_spike_language_neuron_adapter_sparse": bool(
                readiness_checks.get("hecsn_spike_language_neuron_adapter_sparse")
            ),
            "hecsn_spike_language_neuron_adapter_dynamic": bool(
                readiness_checks.get("hecsn_spike_language_neuron_adapter_dynamic")
            ),
        }
        checkpoint_path = Path(str(self._checkpoint_path_str))
        rollback_readiness = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_available": checkpoint_path.exists(),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_metadata_available": bool(self._metadata),
            "restore_endpoint_available": True,
            "rollback_policy_required": True,
        }
        snn_language_plasticity_path = {
            "surface": "snn_language_plasticity_path_evidence.v1",
            "artifact_kind": "terminus_snn_language_plasticity_path_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "requires_device_evidence": True,
            "requires_runtime_truth_delta": True,
            "requires_rollback_evidence": True,
            "rollback_readiness": rollback_readiness,
            "latest_gate": "snn_language_plasticity_live_application_preflight.v1",
            "next_gate": "operator_confirmed_checkpoint_transaction_live_language_plasticity_executor",
            "gates": [
                "snn_language_plasticity_pressure.v1",
                "snn_language_plasticity_trial.v1",
                "snn_language_plasticity_replay_evaluation.v1",
                "snn_language_plasticity_replay_experiment.v1",
                "snn_language_plasticity_application_design.v1",
                "snn_language_plasticity_shadow_application.v1",
                "snn_language_plasticity_live_application_readiness.v1",
                "snn_language_plasticity_live_application_preflight.v1",
            ],
        }
        snn_readout_rollout_server_state_binding = (
            self._snn_readout_rollout_server_state_binding()
        )
        snn_readout_rollout_consolidation_path = (
            self._snn_readout_rollout_consolidation_path()
        )
        snn_readout_emission_review_history = (
            self._snn_readout_emission_review_history()
        )
        snn_readout_emission_replay_design_path = (
            self._snn_readout_emission_replay_design_path()
        )
        snn_readout_applied_synapse_provenance = (
            self._snn_readout_applied_synapse_provenance()
        )
        snn_language_capacity_pressure = self._snn_language_capacity_pressure()
        snn_language_capacity_fixed_boundaries = (
            self._snn_language_capacity_fixed_boundaries()
        )
        snn_applied_replay_lineage_restore_validation = (
            self._snn_applied_replay_lineage_restore_validation()
        )
        sleep_plasticity_proposal = dict(sleep_plasticity_autonomy_proposal or {})
        sleep_plasticity_gate = (
            sleep_plasticity_proposal.get("promotion_gate")
            if isinstance(sleep_plasticity_proposal.get("promotion_gate"), Mapping)
            else {}
        )
        sleep_plasticity_candidate = (
            sleep_plasticity_proposal.get("candidate")
            if isinstance(sleep_plasticity_proposal.get("candidate"), Mapping)
            else {}
        )
        snn_sleep_plasticity_autonomy_gate = {
            "surface": sleep_plasticity_proposal.get("surface"),
            "ready": bool(sleep_plasticity_proposal.get("ready")),
            "advisory": bool(sleep_plasticity_proposal.get("advisory", True)),
            "executable": False,
            "mutates_runtime_state": False,
            "review_ticket_id": sleep_plasticity_candidate.get("review_ticket_id"),
            "recommended_action": sleep_plasticity_candidate.get("recommended_action"),
            "promotion_status": sleep_plasticity_gate.get("status"),
            "next_gate": sleep_plasticity_gate.get("next_gate"),
            "eligible_for_autonomy_planning": bool(
                sleep_plasticity_gate.get("eligible_for_autonomy_planning")
            ),
            "eligible_for_action": False,
            "eligible_for_structural_write": False,
        }
        scheduler_installation_proposal = dict(
            sleep_plasticity_scheduler_installation_autonomy_proposal or {}
        )
        scheduler_installation_gate = (
            scheduler_installation_proposal.get("promotion_gate")
            if isinstance(scheduler_installation_proposal.get("promotion_gate"), Mapping)
            else {}
        )
        scheduler_installation_candidate = (
            scheduler_installation_proposal.get("candidate")
            if isinstance(scheduler_installation_proposal.get("candidate"), Mapping)
            else {}
        )
        snn_sleep_plasticity_scheduler_installation_autonomy_gate = {
            "surface": scheduler_installation_proposal.get("surface"),
            "ready": bool(scheduler_installation_proposal.get("ready")),
            "advisory": bool(scheduler_installation_proposal.get("advisory", True)),
            "executable": False,
            "installs_scheduler": False,
            "registers_timer": False,
            "starts_background_worker": False,
            "mutates_runtime_state": False,
            "scheduler_design_review_ticket_id": scheduler_installation_candidate.get(
                "scheduler_design_review_ticket_id"
            ),
            "scheduler_design_hash": scheduler_installation_candidate.get(
                "scheduler_design_hash"
            ),
            "promotion_status": scheduler_installation_gate.get("status"),
            "next_gate": scheduler_installation_gate.get("next_gate"),
            "eligible_for_autonomy_planning": bool(
                scheduler_installation_gate.get("eligible_for_autonomy_planning")
            ),
            "eligible_for_scheduler_installation_preflight_review": bool(
                scheduler_installation_gate.get(
                    "eligible_for_scheduler_installation_preflight_review"
                )
            ),
            "eligible_for_scheduler_installation": False,
            "eligible_for_action": False,
            "eligible_for_structural_write": False,
        }
        replay_selection_proposal = dict(
            due_cycle_bounded_replay_selection_proposal or {}
        )
        replay_selection_gate = (
            replay_selection_proposal.get("promotion_gate")
            if isinstance(replay_selection_proposal.get("promotion_gate"), Mapping)
            else {}
        )
        replay_selection = (
            replay_selection_proposal.get("selection")
            if isinstance(replay_selection_proposal.get("selection"), Mapping)
            else {}
        )
        snn_due_cycle_bounded_replay_selection_gate = {
            "surface": replay_selection_proposal.get("surface"),
            "ready": bool(replay_selection_proposal.get("ready")),
            "advisory": bool(replay_selection_proposal.get("advisory", True)),
            "executable": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "scheduler_installation_id": replay_selection.get(
                "scheduler_installation_id"
            ),
            "candidate_count": int(replay_selection.get("candidate_count", 0) or 0),
            "promotion_status": replay_selection_gate.get("status"),
            "next_gate": replay_selection_gate.get("next_gate"),
            "eligible_for_operator_sleep_replay_selection_inspection": bool(
                replay_selection_gate.get(
                    "eligible_for_operator_sleep_replay_selection_inspection"
                )
            ),
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_plasticity": False,
            "eligible_for_structural_write": False,
        }
        return {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "recommended_action": recommended_action,
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
                "replay_endpoint": replay_endpoint,
                "source_configuration_hash": source_configuration["configuration_hash"],
                "subcortex_spike_health": subcortex_spike_health,
                "self_repair_gate": self_repair_gate,
                "self_repair_evaluation_gate": self_repair_evaluation_gate,
                "structural_plasticity_gate": structural_plasticity_gate,
                "snn_language_readiness_gate": snn_language_readiness_gate,
                "snn_language_plasticity_path": snn_language_plasticity_path,
                "snn_readout_rollout_server_state_binding": (
                    snn_readout_rollout_server_state_binding
                ),
                "snn_readout_rollout_consolidation_path": (
                    snn_readout_rollout_consolidation_path
                ),
                "snn_readout_emission_review_history": (
                    snn_readout_emission_review_history
                ),
                "snn_readout_emission_replay_design_path": (
                    snn_readout_emission_replay_design_path
                ),
                "snn_readout_applied_synapse_provenance": (
                    snn_readout_applied_synapse_provenance
                ),
                "snn_language_capacity_pressure": snn_language_capacity_pressure,
                "snn_language_capacity_fixed_boundaries": (
                    snn_language_capacity_fixed_boundaries
                ),
                "snn_applied_replay_lineage_restore_validation": (
                    snn_applied_replay_lineage_restore_validation
                ),
                "snn_sleep_plasticity_autonomy_gate": snn_sleep_plasticity_autonomy_gate,
                "snn_sleep_plasticity_scheduler_installation_autonomy_gate": (
                    snn_sleep_plasticity_scheduler_installation_autonomy_gate
                ),
                "snn_due_cycle_bounded_replay_selection_gate": (
                    snn_due_cycle_bounded_replay_selection_gate
                ),
            },
        }

    def _snn_language_capacity_pressure(self) -> dict[str, Any]:
        """Summarize fixed language-neuron capacity pressure without resizing."""

        state = (
            self._language_plasticity_state_fn()
            if self._language_plasticity_state_fn is not None
            else {}
        )
        state = dict(state or {})
        sparse_weights = (
            state.get("sparse_transition_weights")
            if isinstance(state.get("sparse_transition_weights"), Mapping)
            else {}
        )
        provenance = (
            state.get("synapse_provenance_by_key")
            if isinstance(state.get("synapse_provenance_by_key"), Mapping)
            else {}
        )
        capacity_state = self._snn_language_capacity_state(state)
        current_language_neuron_count = int(
            capacity_state["language_neuron_count"]
        )
        configured_sparse_edge_budget = int(capacity_state["sparse_edge_budget"])
        configured_outgoing_fanout_budget = int(
            capacity_state["outgoing_fanout_budget"]
        )
        active_neurons: set[int] = set()
        outgoing_fanout: dict[int, int] = {}
        invalid_synapse_key_count = 0
        for key in dict(sparse_weights).keys():
            parts = str(key).split(":", maxsplit=1)
            try:
                pre_index = int(parts[0])
                post_index = int(parts[1])
            except (IndexError, TypeError, ValueError):
                invalid_synapse_key_count += 1
                continue
            if not (
                0 <= pre_index < current_language_neuron_count
                and 0 <= post_index < current_language_neuron_count
            ):
                invalid_synapse_key_count += 1
                continue
            active_neurons.update((pre_index, post_index))
            outgoing_fanout[pre_index] = int(outgoing_fanout.get(pre_index, 0)) + 1
        sparse_edge_count = len(sparse_weights)
        active_neuron_count = len(active_neurons)
        sparse_budget_occupancy = sparse_edge_count / float(configured_sparse_edge_budget)
        neuron_coverage = active_neuron_count / float(current_language_neuron_count)
        max_outgoing_fanout = max(outgoing_fanout.values(), default=0)
        saturated_source_neuron_count = sum(
            1
            for value in outgoing_fanout.values()
            if int(value) >= configured_outgoing_fanout_budget
        )
        orphan_weight_count = len(
            set(map(str, dict(sparse_weights).keys()))
            - set(map(str, dict(provenance).keys()))
        )
        dangling_provenance_count = len(
            set(map(str, dict(provenance).keys()))
            - set(map(str, dict(sparse_weights).keys()))
        )
        pressure = bool(
            sparse_budget_occupancy >= 0.85
            or neuron_coverage >= 0.90
            or saturated_source_neuron_count > 0
        )
        ready = bool(
            pressure
            and invalid_synapse_key_count == 0
            and orphan_weight_count == 0
            and dangling_provenance_count == 0
        )
        promotion_status = (
            "ready_for_operator_language_capacity_expansion_design_review"
            if ready
            else (
                "waiting_for_capacity_pressure"
                if not pressure
                else "waiting_for_clean_language_capacity_evidence"
            )
        )
        return {
            "surface": "snn_language_capacity_pressure_evidence.v1",
            "artifact_kind": "terminus_snn_language_capacity_pressure_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "runs_replay": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_layers": False,
            "language_capacity": deepcopy(capacity_state),
            "capacity_state_surface": capacity_state["surface"],
            "capacity_state_present": bool(capacity_state["present"]),
            "capacity_state_durable": bool(
                capacity_state["present"]
                and capacity_state["raw_surface"] == _SNN_LANGUAGE_CAPACITY_SURFACE
            ),
            "dynamic_capacity_enabled": False,
            "current_language_neuron_count": current_language_neuron_count,
            "configured_sparse_edge_budget": configured_sparse_edge_budget,
            "configured_outgoing_fanout_budget": configured_outgoing_fanout_budget,
            "sparse_transition_weight_count": sparse_edge_count,
            "active_language_neuron_count": active_neuron_count,
            "sparse_edge_budget_occupancy": sparse_budget_occupancy,
            "active_language_neuron_coverage": neuron_coverage,
            "max_outgoing_fanout": max_outgoing_fanout,
            "saturated_source_neuron_count": saturated_source_neuron_count,
            "invalid_synapse_key_count": invalid_synapse_key_count,
            "orphan_weight_count": orphan_weight_count,
            "dangling_provenance_count": dangling_provenance_count,
            "capacity_pressure_detected": pressure,
            "promotion_status": promotion_status,
            "next_gate": (
                "snn_language_neuron_capacity_expansion_design.v1"
                if ready
                else "collect_snn_language_capacity_pressure"
            ),
            "eligible_for_capacity_expansion_design_review": ready,
            "eligible_for_network_resize": False,
            "eligible_for_neuron_growth": False,
            "eligible_for_layer_growth": False,
            "eligible_for_structural_write": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_capacity_expansion_design_review": ready,
                "eligible_for_network_resize": False,
                "eligible_for_neuron_growth": False,
                "eligible_for_layer_growth": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "capacity_pressure_detected": pressure,
                    "synapse_keys_valid": invalid_synapse_key_count == 0,
                    "no_unprovenanced_weights": orphan_weight_count == 0,
                    "no_dangling_provenance": dangling_provenance_count == 0,
                    "runtime_mutation_absent": True,
                    "network_resize_absent": True,
                    "neuron_growth_absent": True,
                    "layer_growth_absent": True,
                    "checkpoint_write_absent": True,
                    "replay_execution_absent": True,
                    "plasticity_application_absent": True,
                },
            },
        }

    def _snn_language_capacity_fixed_boundaries(self) -> dict[str, Any]:
        """Expose fixed-capacity runtime assumptions before resize review."""

        boundary_inventory = self._snn_language_capacity_boundary_inventory()
        fixed_boundary_count = sum(
            1 for item in boundary_inventory if not item["dynamic_capacity_aware"]
        )
        ready = fixed_boundary_count == 0
        return {
            "surface": "snn_language_capacity_fixed_boundary_evidence.v1",
            "artifact_kind": "terminus_snn_language_capacity_fixed_boundary_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "runs_replay": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "resizes_network": False,
            "adds_neurons": False,
            "adds_layers": False,
            "fixed_boundary_count": fixed_boundary_count,
            "dynamic_capacity_aware_boundary_count": len(boundary_inventory)
            - fixed_boundary_count,
            "boundary_inventory": boundary_inventory,
            "capacity_resize_blocked_by_fixed_boundaries": not ready,
            "promotion_status": "ready_for_capacity_resize_compatibility_audit"
            if ready
            else "blocked_by_fixed_capacity_runtime_boundaries",
            "next_gate": "snn_language_capacity_resize_compatibility_audit.v1"
            if ready
            else "replace_fixed_capacity_runtime_boundaries",
            "eligible_for_capacity_resize_compatibility_audit": ready,
            "eligible_for_capacity_resize_executor": False,
            "eligible_for_network_resize": False,
            "eligible_for_neuron_growth": False,
            "eligible_for_layer_growth": False,
            "eligible_for_structural_write": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": "ready_for_capacity_resize_compatibility_audit"
                if ready
                else "blocked_by_fixed_capacity_runtime_boundaries",
                "eligible_for_capacity_resize_compatibility_audit": ready,
                "eligible_for_capacity_resize_executor": False,
                "eligible_for_network_resize": False,
                "eligible_for_neuron_growth": False,
                "eligible_for_layer_growth": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "all_runtime_boundaries_dynamic_capacity_aware": ready,
                    "fixed_capacity_boundaries_absent": fixed_boundary_count == 0,
                    "runtime_mutation_absent": True,
                    "network_resize_absent": True,
                    "checkpoint_write_absent": True,
                    "replay_execution_absent": True,
                    "plasticity_application_absent": True,
                },
            },
        }

    @staticmethod
    def _snn_language_capacity_boundary_inventory(
        *,
        proposed_language_neuron_count: int | None = None,
        proposed_sparse_edge_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        proposed_neurons = (
            int(proposed_language_neuron_count)
            if proposed_language_neuron_count is not None
            else None
        )
        proposed_edges = (
            int(proposed_sparse_edge_budget)
            if proposed_sparse_edge_budget is not None
            else None
        )
        inventory = [
            {
                "boundary_id": "snn_language_readout_ledger.dense_readout_index_validators",
                "owner": "snn_language_readout_ledger",
                "fixed_language_neuron_count": _SNN_LANGUAGE_NEURON_COUNT,
                "dynamic_capacity_aware": False,
                "boundary_kind": "index_validator",
            },
            {
                "boundary_id": "snn_language_readout_ledger.regeneration_adapter_sparse_index_validators",
                "owner": "snn_language_readout_ledger",
                "capacity_state_aware": True,
                "minimum_language_neuron_count": _SNN_LANGUAGE_NEURON_COUNT,
                "dynamic_capacity_aware": True,
                "boundary_kind": "index_validator",
            },
            {
                "boundary_id": "snn_language_readout_ledger.cuda_dense_tensor_shapes",
                "owner": "snn_language_readout_ledger",
                "fixed_tensor_shape": [
                    _SNN_LANGUAGE_NEURON_COUNT,
                    _SNN_LANGUAGE_NEURON_COUNT,
                ],
                "dynamic_capacity_aware": False,
                "boundary_kind": "tensor_shape",
            },
            {
                "boundary_id": "snn_language_plasticity_executor.neuron_index_validators",
                "owner": "snn_language_plasticity_executor",
                "capacity_state_aware": True,
                "minimum_language_neuron_count": _SNN_LANGUAGE_NEURON_COUNT,
                "dynamic_capacity_aware": True,
                "boundary_kind": "index_validator",
            },
            {
                "boundary_id": "snn_language_readout_ledger.sparse_edge_budget",
                "owner": "snn_language_readout_ledger",
                "fixed_sparse_edge_budget": _SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                "dynamic_capacity_aware": False,
                "boundary_kind": "sparse_budget",
            },
            {
                "boundary_id": "snn_language_plasticity_executor.sparse_edge_budget",
                "owner": "snn_language_plasticity_executor",
                "capacity_state_aware": True,
                "minimum_sparse_edge_budget": _SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                "dynamic_capacity_aware": True,
                "boundary_kind": "sparse_budget",
            },
        ]
        for item in inventory:
            if bool(item["dynamic_capacity_aware"]):
                item["compatible_with_proposed_capacity"] = True
                continue
            if proposed_neurons is not None and (
                "fixed_language_neuron_count" in item or "fixed_tensor_shape" in item
            ):
                item["compatible_with_proposed_capacity"] = (
                    proposed_neurons <= _SNN_LANGUAGE_NEURON_COUNT
                )
            if proposed_edges is not None and "fixed_sparse_edge_budget" in item:
                item["compatible_with_proposed_capacity"] = (
                    proposed_edges <= _SNN_LANGUAGE_SPARSE_EDGE_BUDGET
                )
        return inventory

    @classmethod
    def _snn_language_capacity_state(
        cls,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = (
            state.get("language_capacity")
            if isinstance(state.get("language_capacity"), Mapping)
            else {}
        )
        present = bool(raw)
        return {
            "surface": _SNN_LANGUAGE_CAPACITY_SURFACE,
            "raw_surface": str(raw.get("surface") or "") if present else None,
            "present": present,
            "owned_by_hecsn": True,
            "external_dependency": False,
            "language_neuron_count": cls._positive_capacity_int(
                raw.get("language_neuron_count"),
                default=_SNN_LANGUAGE_NEURON_COUNT,
                minimum=_SNN_LANGUAGE_NEURON_COUNT,
            ),
            "sparse_edge_budget": cls._positive_capacity_int(
                raw.get("sparse_edge_budget"),
                default=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                minimum=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
            ),
            "outgoing_fanout_budget": cls._positive_capacity_int(
                raw.get("outgoing_fanout_budget"),
                default=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
                minimum=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
            ),
            "dynamic_capacity_enabled": False,
            "capacity_expansion_count": cls._positive_capacity_int(
                raw.get("capacity_expansion_count"),
                default=0,
                minimum=0,
            ),
            "resizes_network": False,
            "adds_neurons": False,
            "adds_layers": False,
        }

    @staticmethod
    def _positive_capacity_int(
        value: Any,
        *,
        default: int,
        minimum: int,
    ) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = int(default)
        return max(int(minimum), normalized)

    def _snn_applied_replay_lineage_restore_validation(self) -> dict[str, Any]:
        """Expose restore-side applied replay-lineage validation without mutation."""

        service_state = (
            self._metadata.get("service_state")
            if isinstance(self._metadata.get("service_state"), Mapping)
            else {}
        )
        validation = (
            service_state.get("snn_applied_replay_lineage_restore_validation")
            if isinstance(
                service_state.get("snn_applied_replay_lineage_restore_validation"),
                Mapping,
            )
            else {}
        )
        saved_summary = (
            validation.get("saved_summary")
            if isinstance(validation.get("saved_summary"), Mapping)
            else {}
        )
        restored_summary = (
            validation.get("restored_summary")
            if isinstance(validation.get("restored_summary"), Mapping)
            else {}
        )
        available = (
            validation.get("surface")
            == "snn_applied_replay_lineage_restore_validation.v1"
        )
        saved_summary_available = bool(validation.get("saved_summary_available"))
        counts_match = bool(validation.get("summary_counts_match_restored_state"))
        hash_matches = bool(validation.get("summary_hash_matches_restored_state"))
        summary_matches = bool(validation.get("summary_matches_restored_state"))
        ready = bool(
            available
            and saved_summary_available
            and counts_match
            and hash_matches
            and summary_matches
        )
        return {
            "surface": "snn_applied_replay_lineage_restore_validation_evidence.v1",
            "artifact_kind": (
                "terminus_snn_applied_replay_lineage_restore_validation_evidence"
            ),
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "runs_replay": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "loads_external_checkpoint": False,
            "issues_regeneration_permit": False,
            "available": available,
            "saved_summary_available": saved_summary_available,
            "summary_counts_match_restored_state": counts_match,
            "summary_hash_matches_restored_state": hash_matches,
            "summary_matches_restored_state": summary_matches,
            "saved_lineage_count": int(
                saved_summary.get("applied_replay_lineage_count", 0) or 0
            ),
            "restored_lineage_count": int(
                restored_summary.get("applied_replay_lineage_count", 0) or 0
            ),
            "saved_lineage_material_hash": saved_summary.get(
                "lineage_material_hash"
            ),
            "restored_lineage_material_hash": restored_summary.get(
                "lineage_material_hash"
            ),
            "promotion_status": (
                "ready_for_readout_synapse_provenance_audit"
                if ready
                else "waiting_for_matching_applied_replay_lineage_restore_validation"
            ),
            "next_gate": "snn_language_readout_synapse_provenance_audit.v1",
            "eligible_for_readout_synapse_audit_review": ready,
            "eligible_for_plasticity_application": False,
            "eligible_for_live_replay": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": (
                    "ready_for_readout_synapse_provenance_audit"
                    if ready
                    else "waiting_for_matching_applied_replay_lineage_restore_validation"
                ),
                "eligible_for_readout_synapse_audit_review": ready,
                "eligible_for_plasticity_application": False,
                "eligible_for_live_replay": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "restore_validation_available": available,
                    "saved_summary_available": saved_summary_available,
                    "summary_counts_match_restored_state": counts_match,
                    "summary_hash_matches_restored_state": hash_matches,
                    "summary_matches_restored_state": summary_matches,
                    "runtime_mutation_absent": True,
                    "replay_execution_absent": True,
                    "plasticity_application_absent": True,
                    "checkpoint_write_absent": True,
                    "permit_issuance_absent": True,
                },
            },
        }

    def _snn_readout_emission_replay_design_path(self) -> dict[str, Any]:
        """Summarize reviewed-emission replay-design readiness without text."""

        state = (
            self._readout_ledger_state_fn()
            if self._readout_ledger_state_fn is not None
            else {}
        )
        state = dict(state or {})
        readout_events = [
            dict(item)
            for item in list(state.get("events") or [])
            if isinstance(item, Mapping)
        ]
        review_events = [
            dict(item)
            for item in list(state.get("emission_review_events") or [])
            if isinstance(item, Mapping)
        ]
        readout_by_binding: dict[
            tuple[str, str, str, tuple[str, ...]], dict[str, Any]
        ] = {}
        for event in readout_events:
            labels = tuple(str(value) for value in list(event.get("labels") or []))
            key = (
                str(event.get("prediction_hash") or ""),
                str(event.get("transition_memory_evaluation_hash") or ""),
                str(event.get("persistent_transition_weights_hash") or ""),
                labels,
            )
            if all(key[:3]) and labels:
                readout_by_binding.setdefault(key, event)

        matched: list[dict[str, Any]] = []
        unmatched_count = 0
        for review in review_events:
            labels = tuple(str(value) for value in list(review.get("labels") or []))
            key = (
                str(review.get("prediction_hash") or ""),
                str(review.get("transition_memory_evaluation_hash") or ""),
                str(review.get("persistent_transition_weights_hash") or ""),
                labels,
            )
            readout = readout_by_binding.get(key)
            if readout is None:
                unmatched_count += 1
                continue
            grounding = [
                bool(value) for value in list(readout.get("label_grounding") or [])
            ]
            grounded = bool(grounding) and all(grounding)
            label_hash = hashlib.sha256(
                json.dumps(
                    list(labels),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            matched.append(
                {
                    "emission_review_hash": review.get("emission_review_hash"),
                    "emission_hash": review.get("emission_hash"),
                    "readout_evidence_hash": readout.get("readout_evidence_hash"),
                    "prediction_hash": review.get("prediction_hash"),
                    "transition_memory_evaluation_hash": review.get(
                        "transition_memory_evaluation_hash"
                    ),
                    "persistent_transition_weights_hash": review.get(
                        "persistent_transition_weights_hash"
                    ),
                    "label_hash": label_hash,
                    "grounded": grounded,
                }
            )

        grounded_count = sum(1 for item in matched if bool(item.get("grounded")))
        ready = grounded_count > 0
        latest = matched[0] if matched else {}
        promotion_status = (
            "ready_for_emission_replay_evaluation_design_review"
            if ready
            else (
                "waiting_for_matching_internal_readout_evidence"
                if review_events
                else "waiting_for_reviewed_snn_language_emission"
            )
        )
        return {
            "surface": "snn_readout_emission_replay_design_path_evidence.v1",
            "artifact_kind": (
                "terminus_snn_readout_emission_replay_design_path_evidence"
            ),
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "calls_endpoint": False,
            "records_ledger_event": False,
            "records_replay_context": False,
            "runs_replay": False,
            "writes_checkpoint": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "exposes_raw_text": False,
            "freeform_language_generation": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "emission_review_event_count": len(review_events),
            "internal_readout_evidence_count": len(readout_events),
            "policy_candidate_count": len(matched),
            "design_seed_candidate_count": grounded_count,
            "unmatched_emission_review_count": unmatched_count,
            "latest_emission_review_hash": latest.get("emission_review_hash"),
            "latest_emission_hash": latest.get("emission_hash"),
            "latest_readout_evidence_hash": latest.get("readout_evidence_hash"),
            "latest_prediction_hash": latest.get("prediction_hash"),
            "latest_transition_memory_evaluation_hash": latest.get(
                "transition_memory_evaluation_hash"
            ),
            "latest_persistent_transition_weights_hash": latest.get(
                "persistent_transition_weights_hash"
            ),
            "latest_label_hash": latest.get("label_hash"),
            "requires_device_review_evidence": True,
            "requires_server_computed_mismatch_probe": True,
            "requires_server_computed_plasticity_pressure": True,
            "next_gate": (
                "POST /terminus/snn-language-sequence/readout-emission/"
                "operator-review/replay-evaluation-design"
                if ready
                else (
                    "GET /terminus/snn-language-sequence/readout-emission/"
                    "operator-review/replay-evaluation-policy"
                    if review_events
                    else "snn_language_readout_emission_review_record.v1"
                )
            ),
            "promotion_status": promotion_status,
            "eligible_for_emission_replay_evaluation_design_review": ready,
            "eligible_for_operator_replay_context_review": False,
            "eligible_for_replay_context_recording": False,
            "eligible_for_replay_memory": False,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_emission_replay_evaluation_design_review": ready,
                "eligible_for_operator_replay_context_review": False,
                "eligible_for_replay_context_recording": False,
                "eligible_for_replay_memory": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "reviewed_emission_available": bool(review_events),
                    "matching_internal_readout_evidence_available": bool(matched),
                    "grounded_design_seed_available": ready,
                    "device_review_evidence_required": True,
                    "server_computed_mismatch_probe_required": True,
                    "server_computed_plasticity_pressure_required": True,
                    "raw_text_exposure_absent": True,
                    "runtime_mutation_absent": True,
                    "endpoint_execution_absent": True,
                    "ledger_recording_absent": True,
                    "replay_context_recording_absent": True,
                    "checkpoint_write_absent": True,
                    "replay_memory_promotion_absent": True,
                    "plasticity_application_absent": True,
                    "fact_promotion_absent": True,
                    "action_promotion_absent": True,
                },
            },
        }

    def _snn_readout_emission_review_history(self) -> dict[str, Any]:
        """Summarize reviewed bounded SNN emissions without exposing text or mutating."""

        state = (
            self._readout_ledger_state_fn()
            if self._readout_ledger_state_fn is not None
            else {}
        )
        state = dict(state or {})
        events = [
            dict(item)
            for item in list(state.get("emission_review_events") or [])
            if isinstance(item, Mapping)
        ]
        emission_hashes = sorted(
            {
                str(item.get("emission_hash") or "")
                for item in events
                if str(item.get("emission_hash") or "")
            }
        )
        trajectory_hashes = sorted(
            {
                str(item.get("trajectory_hash") or "")
                for item in events
                if str(item.get("trajectory_hash") or "")
            }
        )
        transition_hashes = sorted(
            {
                str(item.get("persistent_transition_weights_hash") or "")
                for item in events
                if str(item.get("persistent_transition_weights_hash") or "")
            }
        )
        latest = events[0] if events else {}
        review_available = bool(events)
        promotion_status = (
            "ready_for_operator_display_history_inspection"
            if review_available
            else "waiting_for_reviewed_snn_language_emission"
        )
        return {
            "surface": "snn_readout_emission_review_history_evidence.v1",
            "artifact_kind": "terminus_snn_readout_emission_review_history_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "calls_endpoint": False,
            "records_ledger_event": False,
            "runs_replay": False,
            "writes_checkpoint": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "exposes_raw_text": False,
            "freeform_language_generation": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "emission_review_event_count": len(events),
            "total_emission_review_count": int(
                state.get("total_emission_review_count", len(events)) or 0
            ),
            "unique_emission_count": len(emission_hashes),
            "unique_trajectory_count": len(trajectory_hashes),
            "unique_transition_memory_count": len(transition_hashes),
            "latest_emission_reviewed_at": state.get("last_emission_reviewed_at"),
            "latest_emission_review_hash": latest.get("emission_review_hash"),
            "latest_emission_hash": latest.get("emission_hash"),
            "latest_trajectory_hash": latest.get("trajectory_hash"),
            "latest_transition_memory_hash": latest.get(
                "persistent_transition_weights_hash"
            ),
            "next_gate": (
                "GET /terminus/snn-language-sequence/readout-emission/operator-review/history"
                if review_available
                else "snn_language_readout_emission_review_record.v1"
            ),
            "promotion_status": promotion_status,
            "eligible_for_operator_display_history_inspection": review_available,
            "eligible_for_replay_memory": False,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_operator_display_history_inspection": review_available,
                "eligible_for_replay_memory": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "reviewed_emission_available": review_available,
                    "raw_text_exposure_absent": True,
                    "runtime_mutation_absent": True,
                    "endpoint_execution_absent": True,
                    "ledger_recording_absent": True,
                    "checkpoint_write_absent": True,
                    "freeform_language_generation_absent": True,
                    "replay_memory_promotion_absent": True,
                    "plasticity_application_absent": True,
                    "fact_promotion_absent": True,
                    "action_promotion_absent": True,
                },
            },
        }

    def _snn_readout_applied_synapse_provenance(self) -> dict[str, Any]:
        """Summarize applied readout synapse provenance without running audit."""

        state = (
            self._language_plasticity_state_fn()
            if self._language_plasticity_state_fn is not None
            else {}
        )
        state = dict(state or {})
        sparse_weights = (
            state.get("sparse_transition_weights")
            if isinstance(state.get("sparse_transition_weights"), Mapping)
            else {}
        )
        provenance = (
            state.get("synapse_provenance_by_key")
            if isinstance(state.get("synapse_provenance_by_key"), Mapping)
            else {}
        )
        rows = [
            dict(value)
            for value in dict(provenance).values()
            if isinstance(value, Mapping)
        ]
        replay_rows = [
            row for row in rows if str(row.get("provenance_type") or "") == "replay_regeneration"
        ]
        complete_local_edge_rows = 0
        invalid_rollout_step_rows = 0
        replay_artifact_lineage_rows = 0
        complete_replay_artifact_lineage_rows = 0
        for row in replay_rows:
            source_metadata_hash = str(row.get("source_metadata_hash") or "")
            emission_lineage = (
                row.get("emission_lineage")
                if isinstance(row.get("emission_lineage"), Mapping)
                else {}
            )
            lineage_available = bool(source_metadata_hash or emission_lineage)
            lineage_complete = bool(
                not lineage_available
                or (
                    source_metadata_hash
                    and emission_lineage.get("emission_hash")
                    and emission_lineage.get("readout_evidence_hash")
                    and emission_lineage.get("prediction_hash")
                )
            )
            if lineage_available:
                replay_artifact_lineage_rows += 1
            if lineage_available and lineage_complete:
                complete_replay_artifact_lineage_rows += 1
            local = (
                row.get("local_edge_provenance")
                if isinstance(row.get("local_edge_provenance"), Mapping)
                else {}
            )
            try:
                source_step = int(local.get("source_rollout_step_index"))
                target_step = int(local.get("target_rollout_step_index"))
                ordered = target_step > source_step
            except (TypeError, ValueError):
                ordered = False
            complete = bool(
                local.get("source_synapse_id")
                and local.get("source_active_indices_hash")
                and local.get("target_active_indices_hash")
                and ordered
            )
            if complete:
                complete_local_edge_rows += 1
            if not ordered:
                invalid_rollout_step_rows += 1
        missing_local_edge_rows = max(0, len(replay_rows) - complete_local_edge_rows)
        incomplete_lineage_rows = max(
            0,
            replay_artifact_lineage_rows - complete_replay_artifact_lineage_rows,
        )
        orphan_weight_count = len(set(map(str, dict(sparse_weights).keys())) - set(map(str, dict(provenance).keys())))
        dangling_provenance_count = len(
            set(map(str, dict(provenance).keys())) - set(map(str, dict(sparse_weights).keys()))
        )
        restore_validation = self._snn_applied_replay_lineage_restore_validation()
        restore_validation_available = bool(restore_validation.get("available"))
        restore_lineage_matches = bool(
            restore_validation.get("summary_matches_restored_state")
        )
        restore_validation_blocks_audit = bool(
            restore_validation_available and not restore_lineage_matches
        )
        ready = bool(
            provenance
            and orphan_weight_count == 0
            and dangling_provenance_count == 0
            and missing_local_edge_rows == 0
            and invalid_rollout_step_rows == 0
            and incomplete_lineage_rows == 0
            and not restore_validation_blocks_audit
        )
        promotion_status = (
            "ready_for_readout_synapse_provenance_audit"
            if ready
            else (
                "waiting_for_matching_applied_replay_lineage_restore_validation"
                if restore_validation_blocks_audit
                else "waiting_for_complete_applied_synapse_provenance"
            )
        )
        return {
            "surface": "snn_readout_applied_synapse_provenance_evidence.v1",
            "artifact_kind": "terminus_snn_readout_applied_synapse_provenance_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "runs_audit": False,
            "runs_replay": False,
            "calls_endpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "freeform_language_generation": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "writes_checkpoint": False,
            "sparse_transition_weight_count": len(sparse_weights),
            "synapse_provenance_count": len(provenance),
            "replay_regeneration_synapse_count": len(replay_rows),
            "complete_local_edge_provenance_count": complete_local_edge_rows,
            "missing_local_edge_provenance_count": missing_local_edge_rows,
            "invalid_rollout_step_order_count": invalid_rollout_step_rows,
            "replay_artifact_lineage_count": replay_artifact_lineage_rows,
            "complete_replay_artifact_lineage_count": complete_replay_artifact_lineage_rows,
            "incomplete_replay_artifact_lineage_count": incomplete_lineage_rows,
            "orphan_weight_count": orphan_weight_count,
            "dangling_provenance_count": dangling_provenance_count,
            "restore_validation_available": restore_validation_available,
            "restore_lineage_matches_restored_state": restore_lineage_matches,
            "restore_validation_blocks_audit": restore_validation_blocks_audit,
            "promotion_status": promotion_status,
            "next_gate": "snn_language_readout_synapse_provenance_audit.v1",
            "eligible_for_readout_synapse_audit_review": ready,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_readout_synapse_audit_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "synapse_provenance_available": bool(provenance),
                    "no_unprovenanced_weights": orphan_weight_count == 0,
                    "no_dangling_provenance": dangling_provenance_count == 0,
                    "replay_regeneration_local_edge_provenance_complete": (
                        missing_local_edge_rows == 0
                    ),
                    "replay_regeneration_rollout_step_order_valid": (
                        invalid_rollout_step_rows == 0
                    ),
                    "replay_regeneration_artifact_lineage_complete": (
                        incomplete_lineage_rows == 0
                    ),
                    "restore_validation_not_mismatched": (
                        not restore_validation_blocks_audit
                    ),
                    "runtime_mutation_absent": True,
                    "endpoint_execution_absent": True,
                    "audit_execution_absent": True,
                    "checkpoint_write_absent": True,
                    "freeform_language_generation_absent": True,
                },
            },
        }

    def _snn_readout_rollout_consolidation_path(self) -> dict[str, Any]:
        """Summarize rollout rehearsal/consolidation ledger state without executing gates."""

        state = (
            self._readout_ledger_state_fn()
            if self._readout_ledger_state_fn is not None
            else {}
        )
        state = dict(state or {})
        events = [
            dict(item)
            for item in list(state.get("events") or [])
            if isinstance(item, Mapping)
        ]
        rollout_events = [
            dict(item)
            for item in list(state.get("rollout_events") or [])
            if isinstance(item, Mapping)
        ]
        transition_hashes = sorted(
            {
                str(item.get("persistent_transition_weights_hash") or "")
                for item in rollout_events
                if str(item.get("persistent_transition_weights_hash") or "")
            }
        )
        rollout_hashes = sorted(
            {
                str(item.get("rollout_hash") or "")
                for item in rollout_events
                if str(item.get("rollout_hash") or "")
            }
        )
        latest_rollout = rollout_events[0] if rollout_events else {}
        rollout_evidence_available = bool(rollout_events)
        promotion_status = (
            "ready_for_rollout_rehearsal_policy_review"
            if rollout_evidence_available
            else "waiting_for_recorded_rollout_replay_evidence"
        )
        return {
            "surface": "snn_readout_rollout_consolidation_path_evidence.v1",
            "artifact_kind": "terminus_snn_readout_rollout_consolidation_path_evidence",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "executes_rehearsal": False,
            "executes_consolidation": False,
            "runs_live_replay": False,
            "records_ledger_event": False,
            "writes_checkpoint": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "freeform_language_generation": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "event_count": len(events),
            "rollout_event_count": len(rollout_events),
            "total_recorded_count": int(state.get("total_recorded_count", len(events)) or 0),
            "total_rollout_recorded_count": int(
                state.get("total_rollout_recorded_count", len(rollout_events)) or 0
            ),
            "unique_rollout_count": len(rollout_hashes),
            "unique_transition_memory_count": len(transition_hashes),
            "latest_rollout_recorded_at": state.get("last_rollout_recorded_at"),
            "latest_rollout_evidence_hash": latest_rollout.get("rollout_evidence_hash"),
            "latest_rollout_hash": latest_rollout.get("rollout_hash"),
            "latest_transition_memory_hash": latest_rollout.get(
                "persistent_transition_weights_hash"
            ),
            "next_gate": (
                "snn_language_readout_rollout_rehearsal_promotion_policy.v1"
                if rollout_evidence_available
                else "snn_language_readout_rollout_evidence_ledger_record.v1"
            ),
            "promotion_status": promotion_status,
            "eligible_for_rollout_rehearsal_policy_review": rollout_evidence_available,
            "eligible_for_live_replay": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_rollout_rehearsal_policy_review": rollout_evidence_available,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "recorded_rollout_replay_evidence_available": rollout_evidence_available,
                    "runtime_mutation_absent": True,
                    "endpoint_execution_absent": True,
                    "tensor_rehearsal_absent": True,
                    "checkpoint_write_absent": True,
                    "freeform_language_generation_absent": True,
                },
            },
        }

    def _snn_readout_rollout_server_state_binding(self) -> dict[str, Any]:
        """Summarize readout-rollout server-state binding without running rollout."""

        state = (
            self._language_plasticity_state_fn()
            if self._language_plasticity_state_fn is not None
            else {}
        )
        state = dict(state or {})
        sparse_weights = (
            state.get("sparse_transition_weights")
            if isinstance(state.get("sparse_transition_weights"), Mapping)
            else {}
        )
        provenance = (
            state.get("synapse_provenance_by_key")
            if isinstance(state.get("synapse_provenance_by_key"), Mapping)
            else {}
        )
        transition_memory_hash = (
            hashlib.sha256(
                json.dumps(
                    dict(sparse_weights),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if sparse_weights
            else None
        )
        weight_count = len(sparse_weights)
        transition_memory_available = bool(weight_count > 0)
        promotion_status = (
            "ready_for_server_bound_rollout_review"
            if transition_memory_available
            else "waiting_for_server_transition_memory"
        )
        return {
            "surface": "snn_readout_rollout_server_state_binding.v1",
            "artifact_kind": "terminus_snn_readout_rollout_server_state_binding_gate",
            "source": "status_read_model.runtime_truth_contract",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "advisory": True,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "freeform_language_generation": False,
            "loads_external_checkpoint": False,
            "accepts_caller_transition_memory_state": False,
            "requires_server_transition_memory_state": True,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
            "server_transition_memory_available": transition_memory_available,
            "server_transition_memory_hash": transition_memory_hash,
            "server_transition_weight_count": weight_count,
            "server_synapse_provenance_count": len(provenance),
            "current_state_revision": int(self._runtime_state.state_revision),
            "runtime_mutation_absent": True,
            "plasticity_absent": True,
            "checkpoint_write_absent": True,
            "rollout_execution_absent": True,
            "runs_replay": False,
            "records_ledger_event": False,
            "calls_rollout": False,
            "bounded_parameter_route_shape": {
                "rollout_steps": {"min": 1, "max": 12},
                "top_k": {"min": 1, "max": 8},
            },
            "next_gate": "snn_language_readout_rollout_candidate.v1",
            "promotion_status": promotion_status,
            "eligible_for_rollout_execution": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "promotion_gate": {
                "status": promotion_status,
                "eligible_for_bounded_snn_readout_rollout_review": transition_memory_available,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": {
                    "server_transition_memory_available": transition_memory_available,
                    "server_transition_memory_hash_available": transition_memory_hash is not None,
                    "caller_transition_memory_state_absent_or_ignored": True,
                    "bounded_rollout_parameters_enforced": True,
                    "trajectory_evidence_required": True,
                    "external_dependency_absent": True,
                    "runtime_mutation_absent": True,
                },
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
        sleep_plasticity_autonomy_proposal = self._sleep_plasticity_autonomy_proposal()
        scheduler_installation_autonomy_proposal = (
            self._sleep_plasticity_scheduler_installation_autonomy_proposal()
        )
        due_cycle_bounded_replay_selection_proposal = (
            self._due_cycle_bounded_replay_selection_proposal()
        )

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
            "snn_sleep_plasticity_autonomy_proposal": sleep_plasticity_autonomy_proposal,
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal": (
                scheduler_installation_autonomy_proposal
            ),
            "snn_due_cycle_bounded_replay_selection_proposal": (
                due_cycle_bounded_replay_selection_proposal
            ),
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
                sleep_plasticity_autonomy_proposal=sleep_plasticity_autonomy_proposal,
                sleep_plasticity_scheduler_installation_autonomy_proposal=(
                    scheduler_installation_autonomy_proposal
                ),
                due_cycle_bounded_replay_selection_proposal=(
                    due_cycle_bounded_replay_selection_proposal
                ),
            ),
        }

    def _terminus_status_snapshot_locked(self) -> dict[str, Any]:
        """Build the terminus status snapshot. Caller MUST hold self._lock."""
        terminus_runtime = self._brain_runtime_snapshot_fn()
        runtime_mutation = self._runtime_state.mutation_summary()
        replay_dataset_summary = self._replay_dataset_summary_from_runtime(terminus_runtime)
        memory_store = self._trainer.model.memory_store.summary_stats()
        trace_history_size = int(len(self._trace_history))
        sleep_plasticity_autonomy_proposal = self._sleep_plasticity_autonomy_proposal()
        scheduler_installation_autonomy_proposal = (
            self._sleep_plasticity_scheduler_installation_autonomy_proposal()
        )
        due_cycle_bounded_replay_selection_proposal = (
            self._due_cycle_bounded_replay_selection_proposal()
        )
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
            "snn_sleep_plasticity_autonomy_proposal": sleep_plasticity_autonomy_proposal,
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal": (
                scheduler_installation_autonomy_proposal
            ),
            "snn_due_cycle_bounded_replay_selection_proposal": (
                due_cycle_bounded_replay_selection_proposal
            ),
            "runtime_truth": self._runtime_truth_contract_locked(
                terminus_runtime=terminus_runtime,
                memory_store=memory_store,
                replay_dataset_summary=replay_dataset_summary,
                trace_history_size=trace_history_size,
                sleep_plasticity_autonomy_proposal=sleep_plasticity_autonomy_proposal,
                sleep_plasticity_scheduler_installation_autonomy_proposal=(
                    scheduler_installation_autonomy_proposal
                ),
                due_cycle_bounded_replay_selection_proposal=(
                    due_cycle_bounded_replay_selection_proposal
                ),
            ),
        }

    def _telemetry_snapshot_locked(self) -> dict[str, Any]:
        """Build the telemetry dict. Caller MUST hold self._lock."""
        runtime_mutation = self._runtime_state.mutation_summary()
        current_rev = int(runtime_mutation["state_revision"])
        if self._cached_telemetry is not None and self._cached_telemetry_rev == current_rev:
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

    @staticmethod
    def _sha256_json(value: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

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
        the runtime revision hasn't changed, the
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
            payload = self._attach_living_loop_sleep_plasticity_autonomy_proposal(payload)
            payload = self._attach_living_loop_sleep_plasticity_scheduler_installation_autonomy_proposal(
                payload
            )
            payload = self._attach_living_loop_due_cycle_bounded_replay_selection_proposal(
                payload
            )
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

    def _sleep_plasticity_autonomy_proposal(self) -> dict[str, Any] | None:
        if self._sleep_plasticity_autonomy_proposal_fn is None:
            return None
        proposal = self._sleep_plasticity_autonomy_proposal_fn()
        if not isinstance(proposal, Mapping):
            return None
        return {
            **dict(proposal),
            "advisory": True,
            "executable": False,
            "mutates_runtime_state": False,
        }

    def _attach_living_loop_sleep_plasticity_autonomy_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach read-only sleep/plasticity autonomy evidence to living-loop status."""
        proposal = self._sleep_plasticity_autonomy_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        living_loop = dict(enriched.get("living_loop") or {})
        living_loop["snn_sleep_plasticity_autonomy_proposal"] = proposal
        enriched["living_loop"] = living_loop
        return enriched

    def _sleep_plasticity_scheduler_installation_autonomy_proposal(
        self,
    ) -> dict[str, Any] | None:
        callback = self._sleep_plasticity_scheduler_installation_autonomy_proposal_fn
        if callback is None:
            return None
        proposal = callback()
        if not isinstance(proposal, Mapping):
            return None
        return {
            **dict(proposal),
            "advisory": True,
            "executable": False,
            "installs_scheduler": False,
            "registers_timer": False,
            "starts_background_worker": False,
            "mutates_runtime_state": False,
        }

    def _attach_living_loop_sleep_plasticity_scheduler_installation_autonomy_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = self._sleep_plasticity_scheduler_installation_autonomy_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        living_loop = dict(enriched.get("living_loop") or {})
        living_loop[
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
        ] = proposal
        enriched["living_loop"] = living_loop
        return enriched

    def _due_cycle_bounded_replay_selection_proposal(self) -> dict[str, Any] | None:
        callback = self._due_cycle_bounded_replay_selection_proposal_fn
        if callback is None:
            return None
        proposal = callback()
        if not isinstance(proposal, Mapping):
            return None
        return {
            **dict(proposal),
            "advisory": True,
            "executable": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
        }

    def _attach_living_loop_due_cycle_bounded_replay_selection_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = self._due_cycle_bounded_replay_selection_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        living_loop = dict(enriched.get("living_loop") or {})
        living_loop["snn_due_cycle_bounded_replay_selection_proposal"] = proposal
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
            payload = self._attach_policy_self_repair_candidates(payload)
            payload = self._attach_policy_sleep_plasticity_autonomy_proposal(payload)
            payload = self._attach_policy_sleep_plasticity_scheduler_installation_autonomy_proposal(
                payload
            )
            return self._attach_policy_due_cycle_bounded_replay_selection_proposal(
                payload
            )

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

    def _attach_policy_sleep_plasticity_autonomy_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach advisory sleep/plasticity autonomy evidence to policy status."""
        proposal = self._sleep_plasticity_autonomy_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        enriched["snn_sleep_plasticity_autonomy_proposal"] = proposal
        return enriched

    def _attach_policy_sleep_plasticity_scheduler_installation_autonomy_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = self._sleep_plasticity_scheduler_installation_autonomy_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        enriched[
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
        ] = proposal
        return enriched

    def _attach_policy_due_cycle_bounded_replay_selection_proposal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        proposal = self._due_cycle_bounded_replay_selection_proposal()
        if proposal is None:
            return dict(payload)
        enriched = dict(payload)
        enriched["snn_due_cycle_bounded_replay_selection_proposal"] = proposal
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
            return payload
        finally:
            self._lock.release()

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

    def _snn_language_evaluation_surface(self) -> dict[str, Any]:
        """Build a read-only SNN language-adapter evaluation artifact."""
        return build_snn_language_evaluation_surface(
            self.cognitive_signal_state(),
            self._runtime_scope_report_locked(),
        )

    def snn_language_evaluation_surface(self) -> dict[str, Any]:
        """Return isolated evaluation evidence for the local SNN language adapter."""
        result = self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=self._cached_snn_language_evaluation_surface,
            snapshot_fn=self._snn_language_evaluation_surface,
        )
        self._cached_snn_language_evaluation_surface = result
        return result

    def snn_language_adapter_heldout_evaluation(
        self,
        heldout_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate heldout readout slots without mutating runtime state."""

        def snapshot() -> dict[str, Any]:
            runtime_scope = self._runtime_scope_report_locked()
            cuda_runtime = (
                runtime_scope.get("cuda_first_runtime")
                if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
                else {}
            )
            readout_device = device_evidence
            if readout_device is None:
                readout_device = {
                    "device": cuda_runtime.get("tensor_device", "cpu"),
                    "source": "status_read_model.runtime_scope",
                }
            return evaluate_spike_language_adapter_heldout(
                heldout_readout_slot_batches,
                readout_device,
            )

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=snapshot,
        )

    def subcortical_structural_plasticity_isolated_evaluation(
        self,
        pre_snapshot: Mapping[str, Any],
        post_snapshot: Mapping[str, Any],
        *,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate an isolated growth/prune trial without mutating runtime state."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_subcortical_structural_plasticity_isolated(
                pre_snapshot,
                post_snapshot,
                rollback_policy=rollback_policy,
            ),
        )

    def subcortical_structural_mutation_design(
        self,
        isolated_evaluation: Mapping[str, Any],
        *,
        operator_id: str | None = None,
        confirmation: bool = False,
        max_total_edge_delta: int = 16,
    ) -> dict[str, Any]:
        """Build read-only structural mutation design evidence."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_subcortical_structural_mutation_design(
                isolated_evaluation,
                operator_id=operator_id,
                confirmation=confirmation,
                max_total_edge_delta=max_total_edge_delta,
            ),
        )

    def subcortical_structural_mutation_preflight(
        self,
        structural_mutation_design: Mapping[str, Any],
        *,
        expected_state_revision: int,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """Build read-only checkpoint preflight for a structural mutation design."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_subcortical_structural_mutation_preflight(
                structural_mutation_design,
                expected_state_revision=expected_state_revision,
                current_state_revision=int(self._runtime_state.state_revision),
                checkpoint_path=checkpoint_path,
            ),
        )

    def snn_language_training_readiness(
        self,
        heldout_evaluation: Mapping[str, Any],
        *,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Review whether heldout adapter evidence is ready for trainer design."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_snn_language_training_readiness_surface(
                heldout_evaluation,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_trainer_dry_run(
        self,
        training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        validation_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        learning_rate: float = 0.08,
        epochs: int = 2,
    ) -> dict[str, Any]:
        """Run isolated local SNN language trainer evidence without runtime updates."""

        def snapshot() -> dict[str, Any]:
            runtime_scope = self._runtime_scope_report_locked()
            cuda_runtime = (
                runtime_scope.get("cuda_first_runtime")
                if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
                else {}
            )
            device_report = device_evidence
            if device_report is None:
                device_report = {
                    "device": cuda_runtime.get("tensor_device", "cpu"),
                    "source": "status_read_model.runtime_scope",
                }
            return run_spike_language_trainer_dry_run(
                training_readout_slot_batches,
                validation_readout_slot_batches,
                device_report,
                learning_rate=learning_rate,
                epochs=epochs,
            )

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=snapshot,
        )

    def snn_language_trainer_isolated_evaluation(
        self,
        dry_run_report: Mapping[str, Any],
        *,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate dry-run trainer evidence without promoting runtime training."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_trainer_dry_run(
                dry_run_report,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_sequence_prediction_probe(
        self,
        training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        current_readout_slots: Sequence[Mapping[str, Any]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        learning_rate: float = 0.08,
        epochs: int = 2,
        top_k: int = 8,
        persistent_transition_weights: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict sparse next-code evidence without decoding text."""

        def snapshot() -> dict[str, Any]:
            runtime_scope = self._runtime_scope_report_locked()
            cuda_runtime = (
                runtime_scope.get("cuda_first_runtime")
                if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
                else {}
            )
            device_report = device_evidence
            if device_report is None:
                device_report = {
                    "device": cuda_runtime.get("tensor_device", "cpu"),
                    "source": "status_read_model.runtime_scope",
                }
            return predict_spike_language_sequence(
                training_readout_slot_batches,
                current_readout_slots,
                device_report,
                learning_rate=learning_rate,
                epochs=epochs,
                top_k=top_k,
                persistent_transition_weights=persistent_transition_weights,
            )

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=snapshot,
        )

    def snn_language_sequence_mismatch_probe(
        self,
        prediction_report: Mapping[str, Any],
        observed_readout_slots: Sequence[Mapping[str, Any]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare predicted and observed sparse code without learning."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_sequence_mismatch(
                prediction_report,
                observed_readout_slots,
                device_evidence,
            ),
        )

    def snn_language_readout_draft(
        self,
        prediction_report: Mapping[str, Any],
        readout_vocabulary_slots: Sequence[Mapping[str, Any]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        transition_memory_evaluation: Mapping[str, Any] | None = None,
        max_draft_terms: int = 6,
    ) -> dict[str, Any]:
        """Generate a bounded grounded SNN readout draft without mutating runtime."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: generate_snn_language_readout_draft(
                prediction_report,
                readout_vocabulary_slots,
                device_evidence,
                transition_memory_evaluation,
                max_draft_terms=max_draft_terms,
            ),
        )

    def snn_language_readout_emission(
        self,
        readout_draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build an operator-visible bounded SNN readout emission without mutation."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_snn_language_readout_emission(readout_draft),
        )

    def snn_language_readout_rollout_candidate(
        self,
        prediction_report: Mapping[str, Any],
        readout_vocabulary_slots: Sequence[Mapping[str, Any]],
        transition_memory_state: Mapping[str, Any],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        transition_memory_evaluation: Mapping[str, Any] | None = None,
        rollout_steps: int = 4,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Generate a bounded readout rollout candidate without mutating runtime."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: rollout_snn_language_readout_candidate(
                prediction_report=prediction_report,
                readout_vocabulary_slots=readout_vocabulary_slots,
                transition_memory_state=transition_memory_state,
                device_evidence=device_evidence,
                transition_memory_evaluation=transition_memory_evaluation,
                rollout_steps=rollout_steps,
                top_k=top_k,
            ),
        )

    def snn_language_readout_rollout_replay_evaluation(
        self,
        readout_rollout_candidate: Mapping[str, Any],
        *,
        candidate_limit: int = 8,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate bounded rollout replay targets without mutation."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_snn_language_readout_rollout_replay(
                readout_rollout_candidate,
                candidate_limit=candidate_limit,
                device_evidence=device_evidence,
            ),
        )

    def snn_language_transition_memory_prediction_evaluation(
        self,
        training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        evaluation_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
        transition_memory_state: Mapping[str, Any],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        learning_rate: float = 0.08,
        epochs: int = 2,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Evaluate persistent transition-memory utility without mutation."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_snn_language_transition_memory_prediction_evaluation(
                training_readout_slot_batches,
                evaluation_readout_slot_batches,
                transition_memory_state,
                device_evidence,
                learning_rate=learning_rate,
                epochs=epochs,
                top_k=top_k,
            ),
        )

    def snn_language_plasticity_pressure(
        self,
        mismatch_report: Mapping[str, Any],
        *,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert prediction-error evidence into plasticity pressure without learning."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_spike_language_plasticity_pressure(
                mismatch_report,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_trial(
        self,
        pressure_report: Mapping[str, Any],
        *,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate language plasticity pressure without applying learning."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: run_spike_language_plasticity_trial(
                pressure_report,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_replay_evaluation(
        self,
        trial_report: Mapping[str, Any],
        *,
        replay_window: Sequence[Mapping[str, Any]] | None = None,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a language plasticity trial for isolated replay review only."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_plasticity_replay(
                trial_report,
                replay_window=replay_window,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_replay_experiment(
        self,
        replay_evaluation: Mapping[str, Any],
        *,
        replay_sequences: Sequence[Mapping[str, Any]] | None = None,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an isolated language replay experiment without runtime mutation."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: run_spike_language_plasticity_replay_experiment(
                replay_evaluation,
                replay_sequences=replay_sequences,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_application_design(
        self,
        replay_experiment: Mapping[str, Any],
        *,
        application_policy: Mapping[str, Any] | None = None,
        device_evidence: Mapping[str, Any] | None = None,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Design a bounded language plasticity application without applying it."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_spike_language_plasticity_application_design(
                replay_experiment,
                application_policy=application_policy,
                device_evidence=device_evidence,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_shadow_application(
        self,
        application_design: Mapping[str, Any],
        *,
        shadow_delta: Mapping[str, Any] | None = None,
        device_evidence: Mapping[str, Any] | None = None,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify a shadow language plasticity update without applying it."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_plasticity_shadow_application(
                application_design,
                shadow_delta=shadow_delta,
                device_evidence=device_evidence,
                runtime_truth_delta=runtime_truth_delta,
                rollback_policy=rollback_policy,
            ),
        )

    def snn_language_plasticity_shadow_delta(
        self,
        application_design: Mapping[str, Any],
        replay_sequences: Sequence[Mapping[str, Any]],
        *,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure a local language plasticity shadow delta without applying it."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_spike_language_plasticity_shadow_delta(
                application_design,
                replay_sequences,
                device_evidence=device_evidence,
            ),
        )

    def snn_language_plasticity_live_application_readiness(
        self,
        shadow_application: Mapping[str, Any],
        *,
        rollback_readiness: Mapping[str, Any] | None = None,
        operator_approval: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check live language plasticity readiness without applying learning."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_plasticity_live_application_readiness(
                shadow_application,
                rollback_readiness=rollback_readiness,
                operator_approval=operator_approval,
            ),
        )

    def snn_language_plasticity_live_application_preflight(
        self,
        live_application_readiness: Mapping[str, Any],
        *,
        application_target: Mapping[str, Any] | None = None,
        checkpoint_transaction: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Review live language plasticity preflight without applying learning."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: evaluate_spike_language_plasticity_live_application_preflight(
                live_application_readiness,
                application_target=application_target,
                checkpoint_transaction=checkpoint_transaction,
            ),
        )

    def snn_language_capacity_expansion_design(
        self,
        capacity_pressure: Mapping[str, Any],
        *,
        device_evidence: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
        max_neuron_growth_factor: float = 2.0,
    ) -> dict[str, Any]:
        """Design bounded language-neuron capacity expansion without resizing."""

        def _snapshot() -> dict[str, Any]:
            pressure = dict(capacity_pressure)
            gate = (
                pressure.get("promotion_gate")
                if isinstance(pressure.get("promotion_gate"), Mapping)
                else {}
            )
            required_pressure = (
                gate.get("required_evidence")
                if isinstance(gate.get("required_evidence"), Mapping)
                else {}
            )
            device = dict(device_evidence or {})
            rollback = dict(rollback_policy or {})
            current_neurons = int(
                pressure.get("current_language_neuron_count", _SNN_LANGUAGE_NEURON_COUNT)
                or _SNN_LANGUAGE_NEURON_COUNT
            )
            sparse_budget = int(
                pressure.get(
                    "configured_sparse_edge_budget",
                    _SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                )
                or _SNN_LANGUAGE_SPARSE_EDGE_BUDGET
            )
            growth_factor = max(1.0, min(float(max_neuron_growth_factor), 4.0))
            proposed_neurons = max(
                current_neurons + 1,
                min(int(current_neurons * growth_factor), current_neurons * 4),
            )
            proposed_sparse_budget = max(
                sparse_budget + 1,
                min(int(sparse_budget * growth_factor), sparse_budget * 4),
            )
            cuda_available = str(device.get("device") or "").lower().startswith("cuda")
            required = {
                "capacity_pressure_surface_available": pressure.get("surface")
                == "snn_language_capacity_pressure_evidence.v1",
                "capacity_pressure_owned_by_hecsn": bool(pressure.get("owned_by_hecsn")),
                "capacity_pressure_gate_ready": bool(
                    gate.get("eligible_for_capacity_expansion_design_review")
                ),
                "capacity_pressure_detected": bool(
                    pressure.get("capacity_pressure_detected")
                ),
                "capacity_evidence_clean": bool(
                    required_pressure.get("synapse_keys_valid")
                    and required_pressure.get("no_unprovenanced_weights")
                    and required_pressure.get("no_dangling_provenance")
                ),
                "device_evidence_available": bool(device),
                "cuda_device_preferred": cuda_available,
                "rollback_policy_available": bool(rollback.get("available")),
                "checkpoint_required_before_resize": True,
                "runtime_mutation_absent": True,
                "network_resize_absent": True,
            }
            ready = all(required.values())
            design_hash = self._sha256_json(
                {
                    "capacity_pressure_surface": pressure.get("surface"),
                    "current_language_neuron_count": current_neurons,
                    "proposed_language_neuron_count": proposed_neurons,
                    "configured_sparse_edge_budget": sparse_budget,
                    "proposed_sparse_edge_budget": proposed_sparse_budget,
                    "max_neuron_growth_factor": growth_factor,
                    "device": device.get("device"),
                    "rollback_snapshot": rollback.get("snapshot_id"),
                }
            )
            return {
                "artifact_kind": "terminus_snn_language_capacity_expansion_design",
                "surface": "snn_language_neuron_capacity_expansion_design.v1",
                "available": bool(pressure),
                "ready": ready,
                "source": "status_read_model.snn_language_capacity_expansion_design",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "writes_checkpoint": False,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_layers": False,
                "returns_trained_weights": False,
                "capacity_expansion_design_hash": design_hash,
                "capacity_pressure_summary": {
                    "sparse_transition_weight_count": pressure.get(
                        "sparse_transition_weight_count"
                    ),
                    "sparse_edge_budget_occupancy": pressure.get(
                        "sparse_edge_budget_occupancy"
                    ),
                    "active_language_neuron_coverage": pressure.get(
                        "active_language_neuron_coverage"
                    ),
                    "max_outgoing_fanout": pressure.get("max_outgoing_fanout"),
                    "saturated_source_neuron_count": pressure.get(
                        "saturated_source_neuron_count"
                    ),
                },
                "design": {
                    "current_language_neuron_count": current_neurons,
                    "proposed_language_neuron_count": proposed_neurons,
                    "current_sparse_edge_budget": sparse_budget,
                    "proposed_sparse_edge_budget": proposed_sparse_budget,
                    "growth_factor": growth_factor,
                    "requires_cuda_relayout_review": True,
                    "requires_checkpoint_snapshot": True,
                    "requires_restore_validation": True,
                    "preserve_existing_synapse_indices": True,
                    "preserve_existing_synapse_provenance": True,
                },
                "promotion_gate": {
                    "status": "ready_for_operator_capacity_expansion_design_review"
                    if ready
                    else "blocked_missing_language_capacity_expansion_design_evidence",
                    "eligible_for_operator_capacity_expansion_design_review": ready,
                    "eligible_for_checkpoint_backed_capacity_expansion_preflight": False,
                    "eligible_for_network_resize": False,
                    "eligible_for_neuron_growth": False,
                    "eligible_for_layer_growth": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "next_gate": "operator_review_checkpoint_backed_language_capacity_expansion_preflight"
                    if ready
                    else "collect_language_capacity_pressure_device_and_rollback_evidence",
                    "required_evidence": required,
                },
            }

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=_snapshot,
        )

    def snn_language_capacity_expansion_preflight(
        self,
        capacity_expansion_design: Mapping[str, Any],
        *,
        expected_state_revision: int,
        checkpoint_transaction: Mapping[str, Any] | None = None,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Preflight a future capacity expansion without resizing."""

        def _snapshot() -> dict[str, Any]:
            design_artifact = dict(capacity_expansion_design)
            gate = (
                design_artifact.get("promotion_gate")
                if isinstance(design_artifact.get("promotion_gate"), Mapping)
                else {}
            )
            design = (
                design_artifact.get("design")
                if isinstance(design_artifact.get("design"), Mapping)
                else {}
            )
            checkpoint = dict(checkpoint_transaction or {})
            device = dict(device_evidence or {})
            before_revision = int(self._runtime_state.state_revision)
            design_material = {
                "capacity_pressure_surface": "snn_language_capacity_pressure_evidence.v1",
                "current_language_neuron_count": int(
                    design.get("current_language_neuron_count", 0) or 0
                ),
                "proposed_language_neuron_count": int(
                    design.get("proposed_language_neuron_count", 0) or 0
                ),
                "configured_sparse_edge_budget": int(
                    design.get("current_sparse_edge_budget", 0) or 0
                ),
                "proposed_sparse_edge_budget": int(
                    design.get("proposed_sparse_edge_budget", 0) or 0
                ),
                "max_neuron_growth_factor": float(design.get("growth_factor", 1.0) or 1.0),
                "device": device.get("device"),
                "rollback_snapshot": checkpoint.get("snapshot_id"),
            }
            recomputed_design_hash = self._sha256_json(design_material)
            design_hash = str(
                design_artifact.get("capacity_expansion_design_hash") or ""
            )
            cuda_available = str(device.get("device") or "").lower().startswith("cuda")
            required = {
                "design_surface_available": design_artifact.get("surface")
                == "snn_language_neuron_capacity_expansion_design.v1",
                "design_owned_by_hecsn": bool(design_artifact.get("owned_by_hecsn")),
                "design_ready": bool(design_artifact.get("ready")),
                "design_gate_ready": bool(
                    gate.get("eligible_for_operator_capacity_expansion_design_review")
                ),
                "design_hash_available": bool(design_hash),
                "design_hash_recomputed_match": recomputed_design_hash == design_hash,
                "expected_revision_current": int(expected_state_revision)
                == before_revision,
                "checkpoint_transaction_available": bool(checkpoint),
                "checkpoint_path_available": bool(
                    str(checkpoint.get("checkpoint_path") or "").strip()
                ),
                "checkpoint_snapshot_saved": bool(
                    checkpoint.get("pre_expansion_checkpoint_saved")
                    or checkpoint.get("pre_update_checkpoint_saved")
                    or checkpoint.get("checkpoint_saved")
                ),
                "checkpoint_restore_verified": bool(
                    checkpoint.get("restore_verified")
                    or checkpoint.get("pre_expansion_checkpoint_restore_verified")
                    or checkpoint.get("pre_update_checkpoint_restore_verified")
                ),
                "device_evidence_available": bool(device),
                "cuda_relayout_evidence_available": cuda_available,
                "runtime_mutation_absent": True,
                "network_resize_absent": True,
                "checkpoint_write_absent": True,
            }
            ready = all(required.values())
            return {
                "artifact_kind": "terminus_snn_language_capacity_expansion_preflight",
                "surface": "snn_language_neuron_capacity_expansion_preflight.v1",
                "available": bool(design_artifact),
                "ready": ready,
                "source": "status_read_model.snn_language_capacity_expansion_preflight",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "writes_checkpoint": False,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_layers": False,
                "returns_trained_weights": False,
                "expected_state_revision": int(expected_state_revision),
                "capacity_expansion_design_hash": design_hash or None,
                "recomputed_capacity_expansion_design_hash": recomputed_design_hash,
                "checkpoint_transaction": {
                    "checkpoint_path": checkpoint.get("checkpoint_path"),
                    "snapshot_id": checkpoint.get("snapshot_id"),
                    "restore_verified": bool(required["checkpoint_restore_verified"]),
                    "checkpoint_snapshot_saved": bool(
                        required["checkpoint_snapshot_saved"]
                    ),
                },
                "device_evidence": {
                    "device": device.get("device"),
                    "source": device.get("source"),
                    "cuda_relayout_evidence_available": cuda_available,
                },
                "preflight_target": {
                    "current_language_neuron_count": design.get(
                        "current_language_neuron_count"
                    ),
                    "proposed_language_neuron_count": design.get(
                        "proposed_language_neuron_count"
                    ),
                    "current_sparse_edge_budget": design.get(
                        "current_sparse_edge_budget"
                    ),
                    "proposed_sparse_edge_budget": design.get(
                        "proposed_sparse_edge_budget"
                    ),
                    "preserve_existing_synapse_indices": bool(
                        design.get("preserve_existing_synapse_indices")
                    ),
                    "preserve_existing_synapse_provenance": bool(
                        design.get("preserve_existing_synapse_provenance")
                    ),
                },
                "promotion_gate": {
                    "status": "ready_for_operator_capacity_expansion_preflight_review"
                    if ready
                    else "blocked_missing_language_capacity_expansion_preflight_evidence",
                    "eligible_for_operator_capacity_expansion_preflight_review": ready,
                    "eligible_for_checkpoint_backed_capacity_expansion_executor": False,
                    "eligible_for_network_resize": False,
                    "eligible_for_neuron_growth": False,
                    "eligible_for_layer_growth": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "next_gate": "operator_confirmed_checkpoint_backed_language_capacity_expansion_executor"
                    if ready
                    else "collect_capacity_expansion_design_checkpoint_and_cuda_evidence",
                    "required_evidence": required,
                },
            }

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=_snapshot,
        )

    def snn_language_capacity_resize_compatibility_audit(
        self,
        capacity_expansion_preflight: Mapping[str, Any],
        *,
        language_capacity_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Audit fixed-size runtime boundaries before any resize executor exists."""

        def _snapshot() -> dict[str, Any]:
            preflight = dict(capacity_expansion_preflight)
            gate = (
                preflight.get("promotion_gate")
                if isinstance(preflight.get("promotion_gate"), Mapping)
                else {}
            )
            target = (
                preflight.get("preflight_target")
                if isinstance(preflight.get("preflight_target"), Mapping)
                else {}
            )
            capacity_state = self._snn_language_capacity_state(
                {"language_capacity": dict(language_capacity_state or {})}
            )
            current_neurons = int(
                target.get("current_language_neuron_count")
                or capacity_state["language_neuron_count"]
                or _SNN_LANGUAGE_NEURON_COUNT
            )
            proposed_neurons = int(
                target.get("proposed_language_neuron_count")
                or current_neurons
            )
            current_sparse_budget = int(
                target.get("current_sparse_edge_budget")
                or capacity_state["sparse_edge_budget"]
                or _SNN_LANGUAGE_SPARSE_EDGE_BUDGET
            )
            proposed_sparse_budget = int(
                target.get("proposed_sparse_edge_budget")
                or current_sparse_budget
            )
            requested_neuron_growth = proposed_neurons > current_neurons
            requested_sparse_budget_growth = (
                proposed_sparse_budget > current_sparse_budget
            )
            boundary_inventory = self._snn_language_capacity_boundary_inventory(
                proposed_language_neuron_count=proposed_neurons,
                proposed_sparse_edge_budget=proposed_sparse_budget,
            )
            incompatible = [
                item
                for item in boundary_inventory
                if not bool(item["compatible_with_proposed_capacity"])
            ]
            fixed_boundary_count = sum(
                1
                for item in boundary_inventory
                if not bool(item["dynamic_capacity_aware"])
            )
            required = {
                "preflight_surface_available": preflight.get("surface")
                == "snn_language_neuron_capacity_expansion_preflight.v1",
                "preflight_owned_by_hecsn": bool(preflight.get("owned_by_hecsn")),
                "preflight_ready": bool(preflight.get("ready")),
                "preflight_gate_ready": bool(
                    gate.get("eligible_for_operator_capacity_expansion_preflight_review")
                ),
                "capacity_state_surface_available": (
                    capacity_state["surface"] == _SNN_LANGUAGE_CAPACITY_SURFACE
                ),
                "capacity_state_present": bool(capacity_state["present"]),
                "capacity_state_durable": bool(
                    capacity_state["present"]
                    and capacity_state["raw_surface"] == _SNN_LANGUAGE_CAPACITY_SURFACE
                ),
                "requested_neuron_growth_explicit": requested_neuron_growth,
                "requested_sparse_budget_growth_explicit": (
                    requested_sparse_budget_growth
                ),
                "all_runtime_boundaries_dynamic_capacity_aware": (
                    fixed_boundary_count == 0
                ),
                "all_runtime_boundaries_compatible_with_target": not incompatible,
                "runtime_mutation_absent": True,
                "network_resize_absent": True,
                "checkpoint_write_absent": True,
            }
            ready = all(required.values())
            audit_hash = self._sha256_json(
                {
                    "surface": "snn_language_capacity_resize_compatibility_audit.v1",
                    "preflight_hash": preflight.get(
                        "capacity_expansion_design_hash"
                    ),
                    "current_language_neuron_count": current_neurons,
                    "proposed_language_neuron_count": proposed_neurons,
                    "current_sparse_edge_budget": current_sparse_budget,
                    "proposed_sparse_edge_budget": proposed_sparse_budget,
                    "boundary_ids": [
                        item["boundary_id"] for item in boundary_inventory
                    ],
                    "incompatible_boundary_ids": [
                        item["boundary_id"] for item in incompatible
                    ],
                }
            )
            return {
                "artifact_kind": "terminus_snn_language_capacity_resize_compatibility_audit",
                "surface": "snn_language_capacity_resize_compatibility_audit.v1",
                "available": bool(preflight),
                "ready": ready,
                "source": "status_read_model.snn_language_capacity_resize_compatibility_audit",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "writes_checkpoint": False,
                "resizes_network": False,
                "adds_neurons": False,
                "adds_layers": False,
                "returns_trained_weights": False,
                "capacity_resize_compatibility_audit_hash": audit_hash,
                "capacity_target": {
                    "current_language_neuron_count": current_neurons,
                    "proposed_language_neuron_count": proposed_neurons,
                    "current_sparse_edge_budget": current_sparse_budget,
                    "proposed_sparse_edge_budget": proposed_sparse_budget,
                },
                "language_capacity": deepcopy(capacity_state),
                "fixed_boundary_count": fixed_boundary_count,
                "incompatible_boundary_count": len(incompatible),
                "incompatible_boundary_ids": [
                    item["boundary_id"] for item in incompatible
                ],
                "boundary_inventory": boundary_inventory,
                "promotion_gate": {
                    "status": "ready_for_checkpoint_backed_capacity_resize_executor"
                    if ready
                    else "blocked_by_fixed_capacity_runtime_boundaries",
                    "eligible_for_capacity_resize_executor": False,
                    "eligible_for_network_resize": False,
                    "eligible_for_neuron_growth": False,
                    "eligible_for_layer_growth": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_action": False,
                    "next_gate": "replace_fixed_capacity_runtime_boundaries"
                    if not ready
                    else "operator_confirmed_checkpoint_backed_capacity_resize_executor",
                    "required_evidence": required,
                },
            }

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=_snapshot,
        )

    def snn_language_transition_memory_sleep_policy(
        self,
        transition_memory_state: Mapping[str, Any],
        *,
        subcortex_sleep_pressure: Mapping[str, Any] | None = None,
        replay_evidence: Mapping[str, Any] | None = None,
        rollout_regeneration_evidence: Mapping[str, Any] | None = None,
        readout_ledger_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Recommend transition-memory maintenance without mutating runtime state."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_snn_language_transition_memory_sleep_policy(
                transition_memory_state,
                subcortex_sleep_pressure=subcortex_sleep_pressure,
                replay_evidence=replay_evidence,
                rollout_regeneration_evidence=rollout_regeneration_evidence,
                readout_ledger_evidence=readout_ledger_evidence,
            ),
        )

    def snn_language_transition_memory_regeneration_proposal(
        self,
        mismatch_report: Mapping[str, Any],
        transition_memory_state: Mapping[str, Any],
        *,
        replay_evidence: Mapping[str, Any] | None = None,
        locality_radius: int = 2,
        initial_weight: float = 0.02,
        max_new_synapses: int = 8,
    ) -> dict[str, Any]:
        """Propose replay-backed sparse transition regrowth without mutation."""

        return self._read_snapshot(
            fresh_wait_seconds=None,
            cached_snapshot=None,
            snapshot_fn=lambda: build_snn_language_transition_memory_regeneration_proposal(
                mismatch_report,
                transition_memory_state,
                replay_evidence=replay_evidence,
                locality_radius=locality_radius,
                initial_weight=initial_weight,
                max_new_synapses=max_new_synapses,
            ),
        )
