from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from marulho.service.runtime_state import RuntimeState


class DevelopmentalAutonomyScheduler:
    """Advance terminal developmental outcomes without operator approval."""

    def __init__(
        self,
        *,
        lock: RLock,
        runtime_state: RuntimeState,
        plasticity_snapshot: Callable[[], dict[str, Any]],
        language_plasticity_state: Callable[[], Mapping[str, Any]],
        readout_ledger: Any,
        plasticity_executor: Any,
        checkpoint_path: Callable[[], Path],
        save_checkpoint: Callable[[str], Mapping[str, Any]],
        verify_checkpoint_snapshot: Callable[
            [Path, Mapping[str, Any], int], bool
        ],
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._plasticity_snapshot = plasticity_snapshot
        self._language_plasticity_state = language_plasticity_state
        self._readout_ledger = readout_ledger
        self._plasticity_executor = plasticity_executor
        self._checkpoint_path = checkpoint_path
        self._save_checkpoint = save_checkpoint
        self._verify_checkpoint_snapshot = verify_checkpoint_snapshot

    def run_after_tick(
        self,
        *,
        tick_summary: Mapping[str, Any] | None = None,
        learning_executor: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            snapshot = self._plasticity_snapshot()
            developmental = dict(
                snapshot.get(
                    "newborn_neuron_critical_period_state_by_synapse"
                )
                or {}
            )
            open_synapses = sorted(
                key
                for key, value in developmental.items()
                if isinstance(value, Mapping)
                and value.get("current_maturation_state")
                == "critical_period"
            )
            prune_synapses = sorted(
                key
                for key, value in developmental.items()
                if isinstance(value, Mapping)
                and value.get("current_maturation_state")
                == "prune_eligible"
                and not bool(value.get("pruning_applied"))
            )
            if not prune_synapses:
                return {
                    "surface": "developmental_autonomy_cycle.v1",
                    "accepted": True,
                    "mutated": False,
                    "status": (
                        "waiting_for_canonical_synapse_spike_evidence"
                        if open_synapses
                        else "no_terminal_prune_candidates"
                    ),
                    "open_critical_period_synapses": open_synapses,
                    "prune_eligible_synapses": [],
                    "operator_approval_required": False,
                    "fabricated_activity_evidence": False,
                }

            executor_artifact = dict(learning_executor or {})
            if not executor_artifact:
                event = dict(
                    snapshot.get(
                        "last_thought_newborn_neuron_"
                        "critical_period_learning"
                    )
                    or {}
                )
                if not event:
                    return self._blocked(
                        "missing_terminal_learning_event",
                        open_synapses,
                        prune_synapses,
                    )
                committed = str(event.get("committed_checkpoint_path") or "")
                executor_artifact = {
                    "surface": (
                        "snn_language_autonomous_snn_language_thought_"
                        "newborn_neuron_critical_period_learning_executor.v1"
                    ),
                    "accepted": True,
                    "ready": True,
                    "requires_operator_approval": False,
                    "applies_plasticity": True,
                    "resizes_network": False,
                    "adds_neurons": False,
                    "adds_synapses": False,
                    "prunes_network": False,
                    "runs_replay": False,
                    "trains_runtime_model": False,
                    "checkpoint_transaction": {
                        "committed_checkpoint_path": committed,
                        "staged_committed_checkpoint_path": committed,
                        "post_learning_checkpoint_saved": bool(committed),
                        "post_learning_checkpoint_restore_verified": bool(
                            committed
                        ),
                    },
                    "autonomous_snn_language_thought_newborn_neuron_"
                    "critical_period_learning_event": event,
                    "promotion_gate": {
                        "eligible_for_autonomous_snn_language_thought_"
                        "newborn_neuron_critical_period_learning_"
                        "event_review": True
                    },
                }

            revision = int(self._runtime_state.state_revision)
            review = self._readout_ledger.autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review(
                autonomous_snn_language_thought_newborn_neuron_critical_period_learning_executor=executor_artifact,
                plasticity_runtime_state=snapshot,
                expected_state_revision=revision,
            )
            if not bool(review.get("accepted")):
                return self._blocked(
                    "terminal_learning_event_review_failed",
                    open_synapses,
                    prune_synapses,
                    evidence=review,
                )
            outcome = self._readout_ledger.autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(
                autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review=review
            )
            design = self._readout_ledger.autonomous_snn_language_thought_newborn_synapse_pruning_design(
                autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review=outcome
            )
            if not bool(design.get("accepted")):
                return self._blocked(
                    "terminal_pruning_design_failed",
                    open_synapses,
                    prune_synapses,
                    evidence=design,
                )
            checkpoint_source = self._checkpoint_path()
            suffix = checkpoint_source.suffix or ".pt"
            checkpoint = str(
                checkpoint_source.with_name(
                    f"{checkpoint_source.stem}."
                    f"developmental-pruning.rollback{suffix}"
                )
            )
            try:
                saved = self._save_checkpoint(checkpoint)
                saved_path = Path(str(saved.get("path") or checkpoint))
                checkpoint_ready = bool(
                    saved_path.exists()
                    and self._verify_checkpoint_snapshot(
                        saved_path,
                        self._language_plasticity_state(),
                        revision,
                    )
                )
            except Exception:
                checkpoint_ready = False
            if not checkpoint_ready:
                return self._blocked(
                    "terminal_pruning_checkpoint_prepare_failed",
                    open_synapses,
                    prune_synapses,
                )
            preflight = self._readout_ledger.autonomous_snn_language_thought_newborn_synapse_pruning_preflight(
                autonomous_snn_language_thought_newborn_synapse_pruning_design=design,
                expected_state_revision=revision,
                plasticity_runtime_state=snapshot,
                checkpoint_transaction={
                    "pre_pruning_checkpoint_saved": checkpoint_ready,
                    "pre_pruning_checkpoint_restore_verified": (
                        checkpoint_ready
                    ),
                    "checkpoint_path": checkpoint,
                },
                executor_capabilities={
                    "autonomous_snn_language_thought_newborn_"
                    "synapse_pruning_executor": True
                },
            )
            if not bool(preflight.get("accepted")):
                return self._blocked(
                    "terminal_pruning_preflight_failed",
                    open_synapses,
                    prune_synapses,
                    evidence=preflight,
                )
            execution = self._plasticity_executor.apply_autonomous_snn_language_thought_newborn_synapse_pruning(
                autonomous_snn_language_thought_newborn_synapse_pruning_preflight=preflight,
                expected_state_revision=revision,
                checkpoint_path=checkpoint,
            )
            return {
                "surface": "developmental_autonomy_cycle.v1",
                "accepted": bool(execution.get("accepted")),
                "mutated": bool(execution.get("mutates_runtime_state")),
                "status": (
                    "terminal_newborn_synapses_pruned"
                    if execution.get("accepted")
                    else "terminal_pruning_execution_failed"
                ),
                "open_critical_period_synapses": open_synapses,
                "prune_eligible_synapses": prune_synapses,
                "operator_approval_required": False,
                "fabricated_activity_evidence": False,
                "tick_timestamp": (
                    None
                    if tick_summary is None
                    else tick_summary.get("timestamp")
                ),
                "maturation_outcome_review": deepcopy(outcome),
                "pruning_execution": deepcopy(execution),
            }

    @staticmethod
    def _blocked(
        reason: str,
        open_synapses: list[str],
        prune_synapses: list[str],
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "surface": "developmental_autonomy_cycle.v1",
            "accepted": False,
            "mutated": False,
            "status": "blocked",
            "reason": reason,
            "open_critical_period_synapses": open_synapses,
            "prune_eligible_synapses": prune_synapses,
            "operator_approval_required": False,
            "fabricated_activity_evidence": False,
            "evidence": deepcopy(dict(evidence or {})),
        }
