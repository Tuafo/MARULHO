from __future__ import annotations

from pathlib import Path
from threading import RLock

import torch

from marulho.service.developmental_autonomy import (
    DevelopmentalAutonomyScheduler,
)
from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNNLanguagePlasticityApplicationExecutor,
)
from marulho.service.snn_language_readout_ledger import (
    SNNLanguageReadoutEvidenceLedger,
)


class _Ledger:
    def autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review(
        self, **kwargs: object
    ) -> dict[str, object]:
        return {
            "surface": "review",
            "accepted": True,
            "ready": True,
            "review_hash": "r" * 64,
        }

    def autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(
        self, **kwargs: object
    ) -> dict[str, object]:
        return {"surface": "outcome", "accepted": True, "ready": True}

    def autonomous_snn_language_thought_newborn_synapse_pruning_design(
        self, **kwargs: object
    ) -> dict[str, object]:
        return {"surface": "design", "accepted": True, "ready": True}

    def autonomous_snn_language_thought_newborn_synapse_pruning_preflight(
        self, **kwargs: object
    ) -> dict[str, object]:
        return {"surface": "preflight", "accepted": True, "ready": True}


class _Executor:
    def apply_autonomous_snn_language_thought_newborn_synapse_pruning(
        self, **kwargs: object
    ) -> dict[str, object]:
        return {
            "surface": "execution",
            "accepted": True,
            "mutates_runtime_state": True,
        }


def test_scheduler_waits_for_real_synapse_activity_evidence() -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    scheduler = DevelopmentalAutonomyScheduler(
        lock=lock,
        runtime_state=runtime_state,
        plasticity_snapshot=lambda: {
            "newborn_neuron_critical_period_state_by_synapse": {
                "4:64": {
                    "current_maturation_state": "critical_period"
                }
            }
        },
        language_plasticity_state=lambda: {},
        readout_ledger=_Ledger(),
        plasticity_executor=_Executor(),
        checkpoint_path=lambda: Path("brain.pt"),
        save_checkpoint=lambda path: {"path": path},
        verify_checkpoint_snapshot=lambda path, state, revision: True,
    )

    result = scheduler.run_after_tick(
        tick_summary={"timestamp": "2026-06-06T00:00:00+00:00"}
    )

    assert result["mutated"] is False
    assert (
        result["status"]
        == "waiting_for_canonical_synapse_spike_evidence"
    )
    assert result["fabricated_activity_evidence"] is False


def test_scheduler_executes_reviewed_terminal_pruning_without_approval(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    checkpoint = tmp_path / "brain.developmental-pruning.rollback.pt"
    checkpoint.write_bytes(b"checkpoint")
    scheduler = DevelopmentalAutonomyScheduler(
        lock=lock,
        runtime_state=runtime_state,
        plasticity_snapshot=lambda: {
            "newborn_neuron_critical_period_state_by_synapse": {
                "4:64": {
                    "current_maturation_state": "prune_eligible",
                    "pruning_applied": False,
                }
            }
        },
        language_plasticity_state=lambda: {},
        readout_ledger=_Ledger(),
        plasticity_executor=_Executor(),
        checkpoint_path=lambda: tmp_path / "brain.pt",
        save_checkpoint=lambda path: {"path": str(checkpoint)},
        verify_checkpoint_snapshot=lambda path, state, revision: True,
    )
    learning_executor = {
        "surface": "learning-executor",
        "accepted": True,
    }

    result = scheduler.run_after_tick(
        learning_executor=learning_executor
    )

    assert result["accepted"] is True
    assert result["mutated"] is True
    assert result["status"] == "terminal_newborn_synapses_pruned"
    assert result["operator_approval_required"] is False


def test_real_scheduler_prunes_after_terminal_learning_commit(
    tmp_path: Path,
) -> None:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    dense = torch.zeros((66, 66), dtype=torch.float32)
    dense[4, 64] = 0.01
    state = {
        "dense_readout_weights": dense,
        "sparse_transition_weights": {"4:64": 0.01},
        "synapse_provenance_by_key": {
            "4:64": {
                "provenance_type": "newborn_neuron_integration",
                "newborn_integration_synapse_hash": "i" * 64,
            }
        },
        "thought_newborn_neuron_critical_period_learning": {
            "learning_cycle_count": 63,
            "by_synapse": {
                "4:64": {
                    "critical_period_age_cycles": 63,
                    "active_cycle_count": 0,
                    "inactive_cycle_count": 63,
                    "current_maturation_state": "critical_period",
                }
            },
        },
        "language_capacity": {
            "language_neuron_count": 66,
            "sparse_edge_budget": 258,
            "outgoing_fanout_budget": 16,
        },
    }

    def save_checkpoint(path: str | None) -> dict[str, str]:
        target = Path(path or tmp_path / "brain.pt")
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_path / "brain.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: {},
    )
    cycle = {
        "synapse": "4:64",
        "source_neuron_index": 4,
        "target_neuron_index": 64,
        "newborn_integration_synapse_hash": "i" * 64,
        "initial_weight": 0.01,
        "min_weight": 0.0,
        "max_weight": 0.04,
        "max_learning_rate": 0.005,
        "depression_ratio": 0.5,
        "stdp_window_ms": 20.0,
        "target_firing_rate_hz": 4.0,
        "homeostatic_min_firing_rate_hz": 2.0,
        "homeostatic_max_firing_rate_hz": 6.0,
        "critical_period_cycles": 64,
        "critical_period_age_cycles": 63,
        "critical_period_cycles_remaining": 1,
        "minimum_survival_active_cycles": 16,
        "inactivity_prune_cycles": 128,
        "learning_rule": "local_pre_post_timing_with_homeostatic_scaling",
        "survival_rule": "activity_and_prediction_contribution_competition",
        "maturation_states": [
            "critical_period",
            "mature",
            "prune_eligible",
        ],
        "current_maturation_state": "critical_period",
        "critical_period_learning_candidate_hash": "c" * 64,
        "cycle_index": 64,
        "pre_spike_count": 0,
        "post_spike_count": 0,
        "causal_pair_count": 0,
        "anti_causal_pair_count": 0,
        "active_cycle": False,
        "newborn_firing_rate_hz": 0.0,
        "prediction_error": 0.2,
        "current_weight": 0.01,
        "proposed_weight_delta": 0.0,
        "proposed_weight": 0.01,
        "candidate_activity_hash": "a" * 64,
        "weight_update_applied": False,
        "critical_period_age_advanced": False,
        "maturation_decided": False,
        "pruning_applied": False,
    }
    cycle["critical_period_learning_cycle_hash"] = (
        executor._sha256_json(cycle)
    )
    preflight = {
        "surface": (
            "snn_language_autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_preflight.v1"
        ),
        "accepted": True,
        "ready": True,
        "preflight_hash": "p" * 64,
        "requires_operator_approval": False,
        "loads_external_checkpoint": False,
        "runs_replay": False,
        "trains_runtime_model": False,
        "autonomous_snn_language_thought_newborn_neuron_"
        "critical_period_learning_preflight": {
            "thought_newborn_neuron_critical_period_learning_design_hash": (
                "d" * 64
            ),
            "newborn_neuron_integration_event_hash": "e" * 64,
            "expected_state_revision": 0,
            "observation_window_id": "terminal-window",
            "observation_window_hash": "o" * 64,
            "actual_device": "cpu",
            "tensor_is_cuda": False,
            "checkpoint_path": str(tmp_path / "brain.pt"),
            "resolved_cycle_count": 1,
            "resolved_learning_cycles": [cycle],
            "operator_approval_required": False,
        },
        "promotion_gate": {
            "eligible_for_autonomous_snn_language_thought_newborn_neuron_"
            "critical_period_learning_executor": True
        },
    }
    learning = executor.apply_autonomous_snn_language_thought_newborn_neuron_critical_period_learning(
        autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight=preflight,
        expected_state_revision=0,
    )
    scheduler = DevelopmentalAutonomyScheduler(
        lock=lock,
        runtime_state=runtime_state,
        plasticity_snapshot=executor.snapshot,
        language_plasticity_state=lambda: state,
        readout_ledger=ledger,
        plasticity_executor=executor,
        checkpoint_path=lambda: tmp_path / "brain.pt",
        save_checkpoint=lambda path: save_checkpoint(path),
        verify_checkpoint_snapshot=lambda path, value, revision: (
            path.exists() and revision == runtime_state.state_revision
        ),
    )

    result = scheduler.run_after_tick(learning_executor=learning)

    assert learning["accepted"] is True
    assert result["accepted"] is True
    assert result["status"] == "terminal_newborn_synapses_pruned"
    assert runtime_state.state_revision == 2
    assert "4:64" not in state["sparse_transition_weights"]
    assert float(state["dense_readout_weights"][4, 64]) == 0.0
    assert state["pruned_synapse_provenance_by_key"]["4:64"][
        "live_synapse"
    ] is False
