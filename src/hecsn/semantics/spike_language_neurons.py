"""HECSN-owned spike-language neuron adapter evidence.

The adapter consumes the local decoder probe's sparse code and produces
bounded neuron-dynamics evidence. It is intentionally not a generator.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from hecsn.semantics.spike_language_decoder import build_spike_language_decoder_probe


class SpikeLanguageNeuronAdapter:
    """Small PLIF-like adapter over decoder-probe sparse indices."""

    def __init__(self, *, neuron_count: int = 64, threshold: float = 0.65) -> None:
        self.neuron_count = max(16, int(neuron_count))
        self.threshold = max(0.1, float(threshold))

    def evaluate(self, decoder_probe: Mapping[str, Any]) -> dict[str, Any]:
        device_report = (
            decoder_probe.get("device_evidence") if isinstance(decoder_probe.get("device_evidence"), Mapping) else {}
        )
        sparse_code = (
            decoder_probe.get("sparse_code_evidence")
            if isinstance(decoder_probe.get("sparse_code_evidence"), Mapping)
            else {}
        )
        temporal_state = (
            decoder_probe.get("temporal_state_evidence")
            if isinstance(decoder_probe.get("temporal_state_evidence"), Mapping)
            else {}
        )
        support = decoder_probe.get("support_evidence") if isinstance(decoder_probe.get("support_evidence"), Mapping) else {}
        device = _safe_tensor_device(str(device_report.get("tensor_device") or device_report.get("requested_device") or "cpu"))
        active_indices = [
            int(value) % self.neuron_count
            for value in list(sparse_code.get("active_indices") or [])
            if isinstance(value, int)
        ]
        timesteps = max(2 if active_indices else 1, int(_float(temporal_state.get("timestep_count"), 1.0)))
        membrane = torch.zeros(self.neuron_count, device=device)
        spike_trace = torch.zeros((timesteps, self.neuron_count), device=device)
        transition_stride = max(1, int(_float(temporal_state.get("active_transition_count"), 0.0)) + 1)
        for step in range(timesteps):
            membrane.mul_(0.62)
            for index in active_indices:
                routed_index = (index + step * transition_stride) % self.neuron_count
                membrane[routed_index] += 0.72
            spikes = membrane >= self.threshold
            spike_trace[step] = spikes.to(spike_trace.dtype)
            membrane = torch.where(spikes, membrane - self.threshold, membrane)

        active_spikes = int(torch.count_nonzero(spike_trace).item())
        total_slots = int(spike_trace.numel())
        activation_sparsity = 1.0 - (active_spikes / max(1, total_slots))
        grounded = bool(support.get("supported"))
        available = bool(active_indices) and bool(decoder_probe.get("available"))
        return {
            "artifact_kind": "terminus_hecsn_spike_language_neuron_adapter",
            "surface": "snn_language_neuron_adapter_evidence.v1",
            "available": available,
            "source": "semantics.spike_language_neurons",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains": False,
            "mutates_runtime_state": False,
            "executable": False,
            "device_evidence": {
                "requested_device": str(device_report.get("tensor_device") or "unknown"),
                "tensor_device": str(spike_trace.device),
                "cuda_tensor": bool(spike_trace.is_cuda),
                "device_source": device_report.get("device_source"),
            },
            "neuron_dynamics": {
                "neuron_model": "bounded_plif_language_adapter",
                "neuron_count": self.neuron_count,
                "threshold": self.threshold,
                "timestep_count": timesteps,
                "adaptive_timesteps": bool(timesteps > 1),
                "input_active_index_count": len(active_indices),
                "active_spike_count": active_spikes,
                "membrane_norm": float(torch.linalg.vector_norm(membrane).item()),
                "grounded_support": grounded,
            },
            "sparsity_evidence": {
                "activation_sparsity": float(activation_sparsity),
                "target_min_sparsity": 0.85,
                "meets_sparse_activation_floor": bool(activation_sparsity >= 0.85),
            },
            "promotion_constraints": {
                "eligible_for_language_generation": False,
                "requires_training_loop": True,
                "requires_evaluation_dataset": True,
                "requires_grounding_support": True,
                "requires_operator_approval": True,
            },
        }


def build_spike_language_neuron_adapter(decoder_probe: Mapping[str, Any]) -> dict[str, Any]:
    return SpikeLanguageNeuronAdapter().evaluate(decoder_probe)


def evaluate_spike_language_adapter_heldout(
    heldout_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    device_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the local adapter over heldout readout slots without training."""

    device_report = dict(device_evidence or {})
    case_reports: list[dict[str, Any]] = []
    for index, slots in enumerate(heldout_readout_slot_batches):
        readout_slots = [dict(slot) for slot in slots if isinstance(slot, Mapping)]
        decoder_probe = build_spike_language_decoder_probe(
            {
                "readout_slots": readout_slots,
                "device_evidence": device_report,
            }
        )
        adapter = build_spike_language_neuron_adapter(decoder_probe)
        support = decoder_probe.get("support_evidence") if isinstance(decoder_probe.get("support_evidence"), Mapping) else {}
        sparsity = adapter.get("sparsity_evidence") if isinstance(adapter.get("sparsity_evidence"), Mapping) else {}
        dynamics = adapter.get("neuron_dynamics") if isinstance(adapter.get("neuron_dynamics"), Mapping) else {}
        supported = bool(support.get("supported")) and bool(dynamics.get("active_spike_count")) and bool(
            sparsity.get("meets_sparse_activation_floor")
        )
        case_reports.append(
            {
                "case_id": f"heldout_readout_{index}",
                "readout_slot_count": len(readout_slots),
                "grounded_fraction": float(_float(support.get("grounded_fraction"), 0.0)),
                "adapter_active_spike_count": int(_float(dynamics.get("active_spike_count"), 0.0)),
                "adapter_activation_sparsity": float(_float(sparsity.get("activation_sparsity"), 1.0)),
                "tensor_device": (adapter.get("device_evidence") or {}).get("tensor_device")
                if isinstance(adapter.get("device_evidence"), Mapping)
                else None,
                "supported": supported,
            }
        )

    case_count = len(case_reports)
    supported_count = sum(1 for case in case_reports if bool(case.get("supported")))
    mean_grounded = _mean(float(case["grounded_fraction"]) for case in case_reports)
    mean_spikes = _mean(float(case["adapter_active_spike_count"]) for case in case_reports)
    mean_sparsity = _mean(float(case["adapter_activation_sparsity"]) for case in case_reports)
    min_sparsity = min((float(case["adapter_activation_sparsity"]) for case in case_reports), default=1.0)
    ready = bool(case_count) and supported_count == case_count and min_sparsity >= 0.85
    return {
        "artifact_kind": "terminus_snn_language_adapter_heldout_evaluation",
        "surface": "snn_language_adapter_heldout_evaluation.v1",
        "available": bool(case_reports),
        "source": "semantics.spike_language_neurons.heldout_evaluator",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains": False,
        "mutates_runtime_state": False,
        "heldout_summary": {
            "case_count": case_count,
            "supported_case_count": supported_count,
            "unsupported_case_count": max(0, case_count - supported_count),
            "mean_grounded_fraction": mean_grounded,
        },
        "adapter_delta": {
            "mean_active_spike_count": mean_spikes,
            "mean_activation_sparsity": mean_sparsity,
            "min_activation_sparsity": min_sparsity,
            "target_min_activation_sparsity": 0.85,
        },
        "case_reports": case_reports[:8],
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "collect_more_heldout_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_training": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_approved_training_loop_design" if ready else "collect_heldout_grounded_slots",
        },
    }


def run_spike_language_trainer_dry_run(
    training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    validation_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    device_evidence: Mapping[str, Any] | None = None,
    *,
    learning_rate: float = 0.08,
    epochs: int = 2,
) -> dict[str, Any]:
    """Run an isolated local-learning dry run without returning trained weights."""

    device_report = dict(device_evidence or {})
    device = _safe_tensor_device(str(device_report.get("device") or device_report.get("tensor_device") or "cpu"))
    train_patterns = _language_training_patterns(training_readout_slot_batches, device_report)
    validation_patterns = _language_training_patterns(validation_readout_slot_batches, device_report)
    neuron_count = 64
    weights = torch.zeros((neuron_count, neuron_count), device=device)
    lr = max(0.0, min(float(learning_rate), 1.0))
    epoch_count = max(1, min(int(epochs), 8))
    for _ in range(epoch_count):
        for current_indices, target_indices in train_patterns:
            pre = _indices_to_vector(current_indices, neuron_count, device)
            post = _indices_to_vector(target_indices, neuron_count, device)
            weights.mul_(0.995)
            weights.add_(lr * torch.outer(pre, post))
            row_norm = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / row_norm

    validation_reports: list[dict[str, Any]] = []
    for index, (current_indices, target_indices) in enumerate(validation_patterns):
        pre = _indices_to_vector(current_indices, neuron_count, device)
        logits = pre @ weights
        active_count = max(1, min(len(target_indices), 8))
        prediction_indices = set(torch.topk(logits, k=active_count).indices.detach().cpu().tolist())
        target_set = {int(value) % neuron_count for value in target_indices}
        hit_count = len(prediction_indices.intersection(target_set))
        validation_reports.append(
            {
                "case_id": f"validation_transition_{index}",
                "target_active_index_count": len(target_set),
                "predicted_active_index_count": len(prediction_indices),
                "hit_count": hit_count,
                "support": hit_count / max(1, len(target_set)),
            }
        )

    support_mean = _mean(float(case["support"]) for case in validation_reports)
    nonzero_weight_count = int(torch.count_nonzero(weights).item())
    total_weight_count = int(weights.numel())
    weight_sparsity = 1.0 - (nonzero_weight_count / max(1, total_weight_count))
    ready = bool(train_patterns and validation_reports) and support_mean > 0.0 and weight_sparsity >= 0.85
    return {
        "artifact_kind": "terminus_snn_language_trainer_dry_run",
        "surface": "snn_language_trainer_dry_run.v1",
        "available": bool(train_patterns and validation_patterns),
        "source": "semantics.spike_language_neurons.trainer_dry_run",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "tensor_device": str(weights.device),
            "cuda_tensor": bool(weights.is_cuda),
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "training_rule": {
            "rule": "local_hebbian_outer_product_with_row_normalization",
            "learning_rate": lr,
            "epochs": epoch_count,
            "training_transition_count": len(train_patterns),
            "validation_transition_count": len(validation_reports),
        },
        "weight_evidence": {
            "nonzero_weight_count": nonzero_weight_count,
            "weight_sparsity": float(weight_sparsity),
            "target_min_weight_sparsity": 0.85,
        },
        "validation_summary": {
            "case_count": len(validation_reports),
            "mean_transition_support": support_mean,
            "supported_case_count": sum(1 for case in validation_reports if float(case["support"]) > 0.0),
        },
        "validation_reports": validation_reports[:8],
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "collect_more_sequence_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_trainer_promotion": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_approved_isolated_snn_language_trainer_evaluation"
            if ready
            else "collect_grounded_sequence_transitions",
        },
    }


def evaluate_spike_language_trainer_dry_run(
    dry_run_report: Mapping[str, Any],
    *,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate an isolated trainer dry run without promoting runtime training."""

    report = dict(dry_run_report)
    validation = (
        report.get("validation_summary")
        if isinstance(report.get("validation_summary"), Mapping)
        else {}
    )
    weights = report.get("weight_evidence") if isinstance(report.get("weight_evidence"), Mapping) else {}
    device = report.get("device_evidence") if isinstance(report.get("device_evidence"), Mapping) else {}
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    required = {
        "dry_run_available": bool(report.get("available")),
        "dry_run_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "external_dependency_absent": not bool(report.get("external_dependency")),
        "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
        "generation_absent": not bool(report.get("generates_text")),
        "runtime_training_absent": not bool(report.get("trains_runtime_model")),
        "trained_weights_absent": not bool(report.get("returns_trained_weights")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "validation_support_positive": _float(validation.get("mean_transition_support"), 0.0) > 0.0,
        "validation_cases_available": int(_float(validation.get("case_count"), 0.0)) > 0,
        "weight_sparsity_floor_met": _float(weights.get("weight_sparsity"), 0.0)
        >= _float(weights.get("target_min_weight_sparsity"), 0.85),
        "device_evidence_available": bool(device.get("tensor_device")),
        "dry_run_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_trainer_isolated_evaluation",
        "surface": "snn_language_trainer_isolated_evaluation.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.trainer_evaluation",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "promotes_runtime_trainer": False,
        "mutates_runtime_state": False,
        "dry_run_surface": report.get("surface"),
        "validation_summary": {
            "case_count": int(_float(validation.get("case_count"), 0.0)),
            "mean_transition_support": _float(validation.get("mean_transition_support"), 0.0),
            "supported_case_count": int(_float(validation.get("supported_case_count"), 0.0)),
        },
        "weight_evidence": {
            "weight_sparsity": _float(weights.get("weight_sparsity"), 0.0),
            "target_min_weight_sparsity": _float(weights.get("target_min_weight_sparsity"), 0.85),
            "nonzero_weight_count": int(_float(weights.get("nonzero_weight_count"), 0.0)),
        },
        "device_evidence": dict(device),
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_trainer_evaluation_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_trainer_promotion": False,
            "eligible_for_training_loop_design": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_local_snn_language_trainer_design"
            if ready
            else "collect_trainer_evaluation_evidence",
            "required_evidence": required,
        },
        "success_evidence": [
            "trainer_dry_run_report",
            "validation_transition_support",
            "weight_sparsity_evidence",
            "device_evidence_report",
            "runtime_truth_delta",
            "rollback_policy",
        ],
    }


def predict_spike_language_sequence(
    training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    current_readout_slots: Sequence[Mapping[str, Any]],
    device_evidence: Mapping[str, Any] | None = None,
    *,
    learning_rate: float = 0.08,
    epochs: int = 2,
    top_k: int = 8,
    persistent_transition_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict the next sparse spike-code indices without decoding text."""

    device_report = dict(device_evidence or {})
    device = _safe_tensor_device(str(device_report.get("device") or device_report.get("tensor_device") or "cpu"))
    train_patterns = _language_training_patterns(training_readout_slot_batches, device_report)
    current_probe = build_spike_language_decoder_probe(
        {
            "readout_slots": [dict(slot) for slot in current_readout_slots if isinstance(slot, Mapping)],
            "device_evidence": device_report,
        }
    )
    current_sparse = (
        current_probe.get("sparse_code_evidence")
        if isinstance(current_probe.get("sparse_code_evidence"), Mapping)
        else {}
    )
    current_indices = [
        int(value)
        for value in list(current_sparse.get("active_indices") or [])
        if isinstance(value, int)
    ]
    neuron_count = 64
    weights = _train_sequence_transition_weights(
        train_patterns,
        neuron_count,
        device,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    persistent_weights = _persistent_transition_weight_tensor(
        persistent_transition_weights,
        neuron_count,
        device,
    )
    transition_memory_hash = _sha256_json(dict(persistent_transition_weights or {}))
    training_sequence_hash = _sha256_json(
        [[dict(slot) for slot in batch if isinstance(slot, Mapping)] for batch in training_readout_slot_batches]
    )
    persistent_nonzero_weight_count = int(torch.count_nonzero(persistent_weights).item())
    if persistent_nonzero_weight_count > 0:
        weights = weights + persistent_weights
    current_vector = _indices_to_vector(current_indices, neuron_count, device)
    logits = current_vector @ weights
    persistent_logits = current_vector @ persistent_weights
    requested_k = max(1, min(int(top_k), 16, neuron_count))
    predicted = torch.topk(logits, k=requested_k)
    predicted_indices = [int(value) for value in predicted.indices.detach().cpu().tolist()]
    predicted_strengths = [float(value) for value in predicted.values.detach().cpu().tolist()]
    nonzero_weight_count = int(torch.count_nonzero(weights).item())
    total_weight_count = int(weights.numel())
    weight_sparsity = 1.0 - (nonzero_weight_count / max(1, total_weight_count))
    support_strength = float(sum(max(0.0, value) for value in predicted_strengths))
    persistent_support_strength = float(torch.clamp(persistent_logits, min=0.0).sum().item())
    available = bool(train_patterns and current_indices)
    current_sparse_hash = _sha256_json(current_indices[:16])
    current_readout_hash = _sha256_json([dict(slot) for slot in current_readout_slots if isinstance(slot, Mapping)])
    current_sparse_payload = {
        "active_index_count": len(current_indices),
        "active_indices": current_indices[:16],
        "active_indices_hash": current_sparse_hash,
        "mean_sparsity": _float(current_sparse.get("mean_sparsity"), 1.0),
    }
    prediction_payload = {
        "predicted_sparse_indices": predicted_indices,
        "predicted_sparse_strengths": predicted_strengths,
        "support_strength": support_strength,
        "top_k": requested_k,
    }
    persistent_payload = {
        "surface": "snn_language_persistent_transition_evidence.v1",
        "available": persistent_nonzero_weight_count > 0,
        "owned_by_hecsn": True,
        "external_dependency": False,
        "weight_count": persistent_nonzero_weight_count,
        "support_strength": persistent_support_strength,
        "influenced_prediction": persistent_support_strength > 0.0,
        "source": "service.snn_language_plasticity_state"
        if persistent_nonzero_weight_count > 0
        else "none",
        "transition_memory_hash": transition_memory_hash,
        "persistent_transition_weights_hash": transition_memory_hash,
    }
    training_payload = {
        "rule": "local_hebbian_outer_product_with_row_normalization",
        "training_transition_count": len(train_patterns),
        "learning_rate": max(0.0, min(float(learning_rate), 1.0)),
        "epochs": max(1, min(int(epochs), 8)),
        "weight_sparsity": float(weight_sparsity),
        "target_min_weight_sparsity": 0.85,
    }
    prediction_hash = _sha256_json(
        {
            "current_sparse_code": current_sparse_payload,
            "prediction": prediction_payload,
            "training_evidence": training_payload,
            "training_window_hash": training_sequence_hash,
            "current_readout_hash": current_readout_hash,
            "persistent_transition_weights_hash": transition_memory_hash,
        }
    )
    ready = available and support_strength > 0.0 and weight_sparsity >= 0.85
    return {
        "artifact_kind": "terminus_snn_language_sequence_prediction_probe",
        "surface": "snn_language_sequence_prediction_probe.v1",
        "available": available,
        "source": "semantics.spike_language_neurons.sequence_prediction_probe",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "tensor_device": str(weights.device),
            "cuda_tensor": bool(weights.is_cuda),
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "current_sparse_code": current_sparse_payload,
        "prediction": prediction_payload,
        "persistent_transition_evidence": persistent_payload,
        "provenance_evidence": {
            "prediction_hash": prediction_hash,
            "prediction_id": f"snn-seq-pred:{prediction_hash[:16]}",
            "training_sequence_hash": training_sequence_hash,
            "training_window_hash": training_sequence_hash,
            "current_readout_hash": current_readout_hash,
            "current_sparse_code_hash": current_sparse_hash,
            "transition_memory_hash": transition_memory_hash,
            "persistent_transition_weights_hash": transition_memory_hash,
            "hash_algorithm": "sha256_canonical_json",
        },
        "training_evidence": training_payload,
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "collect_more_sequence_context",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_approved_snn_language_prediction_evaluation"
            if ready
            else "collect_grounded_sequence_context",
        },
    }


def generate_snn_language_readout_draft(
    prediction_report: Mapping[str, Any],
    readout_vocabulary_slots: Sequence[Mapping[str, Any]],
    device_evidence: Mapping[str, Any] | None = None,
    transition_memory_evaluation: Mapping[str, Any] | None = None,
    *,
    max_draft_terms: int = 6,
) -> dict[str, Any]:
    """Generate a bounded grounded readout draft from sparse SNN prediction evidence."""

    report = dict(prediction_report)
    prediction = report.get("prediction") if isinstance(report.get("prediction"), Mapping) else {}
    persistent = (
        report.get("persistent_transition_evidence")
        if isinstance(report.get("persistent_transition_evidence"), Mapping)
        else {}
    )
    prediction_provenance = (
        report.get("provenance_evidence")
        if isinstance(report.get("provenance_evidence"), Mapping)
        else {}
    )
    device_report = dict(device_evidence or report.get("device_evidence") or {})
    predicted_indices = [
        int(value)
        for value in list(prediction.get("predicted_sparse_indices") or [])
        if isinstance(value, int)
    ]
    predicted_strengths = [
        float(value)
        for value in list(prediction.get("predicted_sparse_strengths") or [])
        if isinstance(value, (int, float))
    ]
    strength_by_index = {
        int(index) % 32: float(predicted_strengths[position])
        if position < len(predicted_strengths)
        else 0.0
        for position, index in enumerate(predicted_indices)
    }
    candidates: list[dict[str, Any]] = []
    grounded_count = 0
    for slot in [dict(item) for item in readout_vocabulary_slots if isinstance(item, Mapping)][:32]:
        label = _text(slot.get("label"))
        if not label:
            continue
        if bool(slot.get("grounded")):
            grounded_count += 1
        probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [slot],
                "device_evidence": device_report,
            }
        )
        sparse = probe.get("sparse_code_evidence") if isinstance(probe.get("sparse_code_evidence"), Mapping) else {}
        active = [
            int(value) % 32
            for value in list(sparse.get("active_indices") or [])
            if isinstance(value, int)
        ]
        matched = [index for index in active if index in strength_by_index]
        if not matched:
            continue
        score = sum(max(0.0, strength_by_index[index]) for index in matched)
        candidates.append(
            {
                "label": label,
                "grounded": bool(slot.get("grounded")),
                "pressure_band": _text(slot.get("pressure_band")),
                "matched_sparse_indices": matched[:8],
                "support_strength": float(score),
            }
        )
    candidates.sort(key=lambda item: (-float(item["support_strength"]), item["label"]))
    requested_terms = max(1, min(int(max_draft_terms), 12))
    selected = candidates[:requested_terms]
    draft_text = " ".join(str(item["label"]) for item in selected)
    vocabulary_count = len([slot for slot in readout_vocabulary_slots if isinstance(slot, Mapping)])
    grounded_fraction = grounded_count / max(1, vocabulary_count)
    persistent_ready = bool(persistent.get("available")) and bool(persistent.get("influenced_prediction"))
    evaluation = (
        dict(transition_memory_evaluation)
        if isinstance(transition_memory_evaluation, Mapping)
        else {}
    )
    evaluation_summary = (
        evaluation.get("evaluation_summary")
        if isinstance(evaluation.get("evaluation_summary"), Mapping)
        else {}
    )
    evaluation_provenance = (
        evaluation.get("provenance_evidence")
        if isinstance(evaluation.get("provenance_evidence"), Mapping)
        else {}
    )
    evaluation_gate = (
        evaluation.get("promotion_gate")
        if isinstance(evaluation.get("promotion_gate"), Mapping)
        else {}
    )
    evaluation_available = bool(evaluation)
    prediction_hash = str(prediction_provenance.get("prediction_hash") or "")
    evaluation_hash = str(evaluation_provenance.get("evaluation_hash") or "")
    prediction_training_hash = str(
        prediction_provenance.get("training_window_hash")
        or prediction_provenance.get("training_sequence_hash")
        or ""
    )
    evaluation_training_hash = str(
        evaluation_provenance.get("training_window_hash")
        or evaluation_provenance.get("training_sequence_hash")
        or ""
    )
    prediction_transition_hash = str(
        prediction_provenance.get("persistent_transition_weights_hash")
        or prediction_provenance.get("transition_memory_hash")
        or persistent.get("persistent_transition_weights_hash")
        or persistent.get("transition_memory_hash")
        or ""
    )
    evaluation_transition_hash = str(
        evaluation_provenance.get("persistent_transition_weights_hash")
        or evaluation_provenance.get("transition_memory_hash")
        or ""
    )
    prediction_current_readout_hash = str(prediction_provenance.get("current_readout_hash") or "")
    evaluated_prediction_hashes = {
        str(value)
        for value in list(evaluation_provenance.get("evaluated_prediction_hashes") or [])
        if str(value)
    }
    evaluated_current_readout_hashes = {
        str(value)
        for value in list(evaluation_provenance.get("evaluated_current_readout_hashes") or [])
        if str(value)
    }
    provenance_match = (
        bool(prediction_hash)
        and bool(evaluation_hash)
        and prediction_hash in evaluated_prediction_hashes
        and bool(prediction_training_hash)
        and prediction_training_hash == evaluation_training_hash
        and bool(prediction_transition_hash)
        and prediction_transition_hash == evaluation_transition_hash
        and bool(prediction_current_readout_hash)
        and prediction_current_readout_hash in evaluated_current_readout_hashes
    )
    evaluation_non_worsening = (
        evaluation_available
        and
        _float(evaluation_summary.get("mean_mismatch_delta"), -1.0) >= -1e-9
        and int(evaluation_summary.get("worsened_sequence_count") or 0) == 0
    )
    prediction_non_mutating = not bool(report.get("mutates_runtime_state"))
    prediction_external_dependency_absent = not bool(report.get("external_dependency"))
    evaluation_ready = (
        evaluation.get("surface") == "snn_language_transition_memory_prediction_evaluation.v1"
        and bool(evaluation.get("owned_by_hecsn"))
        and not bool(evaluation.get("external_dependency"))
        and not bool(evaluation.get("mutates_runtime_state"))
        and bool(evaluation_gate.get("eligible_for_bounded_readout_generation_review"))
        and int(evaluation_summary.get("persistent_transition_weight_count") or 0) > 0
        and int(evaluation_summary.get("evaluation_pair_count") or 0) > 0
        and int(evaluation_summary.get("influenced_prediction_count") or 0) > 0
        and provenance_match
        and evaluation_non_worsening
    )
    ready = (
        bool(draft_text)
        and grounded_fraction >= 0.5
        and persistent_ready
        and bool(report.get("owned_by_hecsn"))
        and prediction_external_dependency_absent
        and prediction_non_mutating
        and evaluation_ready
    )
    blocked_status = (
        "collect_persistent_grounded_sparse_support"
        if not (
            bool(draft_text)
            and grounded_fraction >= 0.5
            and persistent_ready
            and bool(report.get("owned_by_hecsn"))
            and prediction_external_dependency_absent
            and prediction_non_mutating
        )
        else "collect_transition_memory_prediction_evaluation"
        if not evaluation_available
        else "blocked_transition_memory_prediction_evaluation"
    )
    blocked_next_gate = (
        "collect_persistent_transition_and_grounded_vocabulary"
        if blocked_status == "collect_persistent_grounded_sparse_support"
        else "collect_grounded_transition_memory_evaluation_window"
    )
    return {
        "artifact_kind": "terminus_snn_language_readout_draft",
        "surface": "snn_language_readout_draft.v1",
        "available": bool(draft_text),
        "source": "semantics.spike_language_neurons.readout_draft",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": bool(draft_text),
        "decodes_text": bool(draft_text),
        "generation_scope": "bounded_grounded_readout_label_draft",
        "freeform_language_generation": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "prediction_surface": report.get("surface"),
        "draft": {
            "text": draft_text,
            "term_count": len(selected),
            "max_terms": requested_terms,
            "labels": [str(item["label"]) for item in selected],
        },
        "sparse_decode_evidence": {
            "predicted_index_count": len(predicted_indices),
            "matched_candidate_count": len(candidates),
            "selected_candidate_count": len(selected),
            "candidate_matches": selected,
        },
        "grounding_evidence": {
            "vocabulary_slot_count": vocabulary_count,
            "grounded_slot_count": grounded_count,
            "grounded_fraction": float(grounded_fraction),
            "requires_grounded_fraction": 0.5,
        },
        "persistent_transition_evidence": {
            "available": bool(persistent.get("available")),
            "influenced_prediction": bool(persistent.get("influenced_prediction")),
            "support_strength": _float(persistent.get("support_strength"), 0.0),
            "source": persistent.get("source"),
        },
        "transition_memory_evaluation_evidence": {
            "available": evaluation_available,
            "surface": evaluation.get("surface"),
            "owned_by_hecsn": bool(evaluation.get("owned_by_hecsn")),
            "non_worsening": evaluation_non_worsening,
            "review_ready": evaluation_ready,
            "provenance_match": provenance_match,
            "prediction_hash": prediction_hash or None,
            "transition_memory_evaluation_hash": evaluation_hash or None,
            "persistent_transition_weights_hash": prediction_transition_hash or None,
            "evaluation_pair_count": int(evaluation_summary.get("evaluation_pair_count") or 0),
            "influenced_prediction_count": int(evaluation_summary.get("influenced_prediction_count") or 0),
            "persistent_transition_weight_count": int(
                evaluation_summary.get("persistent_transition_weight_count") or 0
            ),
            "mean_mismatch_delta": _float(evaluation_summary.get("mean_mismatch_delta"), 0.0),
            "worsened_sequence_count": int(evaluation_summary.get("worsened_sequence_count") or 0),
        },
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else blocked_status,
            "eligible_for_bounded_readout_generation": ready,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_review_snn_language_readout_draft"
            if ready
            else blocked_next_gate,
            "required_evidence": {
                "prediction_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
                "persistent_transition_influenced_prediction": persistent_ready,
                "transition_memory_prediction_evaluation_ready": evaluation_ready,
                "transition_memory_prediction_non_worsening": evaluation_non_worsening,
                "transition_memory_prediction_influenced": int(
                    evaluation_summary.get("influenced_prediction_count") or 0
                )
                > 0,
                "transition_memory_prediction_provenance_match": provenance_match,
                "grounded_vocabulary_available": grounded_fraction >= 0.5,
                "draft_text_available": bool(draft_text),
                "external_dependency_absent": prediction_external_dependency_absent,
                "runtime_mutation_absent": prediction_non_mutating,
            },
        },
    }


def rollout_snn_language_readout_candidate(
    prediction_report: Mapping[str, Any],
    readout_vocabulary_slots: Sequence[Mapping[str, Any]],
    transition_memory_state: Mapping[str, Any],
    device_evidence: Mapping[str, Any] | None = None,
    transition_memory_evaluation: Mapping[str, Any] | None = None,
    *,
    rollout_steps: int = 4,
    top_k: int = 4,
) -> dict[str, Any]:
    """Roll HECSN sparse transition memory forward into bounded grounded labels."""

    report = dict(prediction_report)
    state = dict(transition_memory_state or {})
    persistent_weights = (
        state.get("sparse_transition_weights")
        if isinstance(state.get("sparse_transition_weights"), Mapping)
        else {}
    )
    persistent_weights_hash = _sha256_json(dict(persistent_weights))
    device_report = dict(device_evidence or report.get("device_evidence") or {})
    device = _safe_tensor_device(str(device_report.get("device") or device_report.get("tensor_device") or "cpu"))
    memory_tensor = _persistent_transition_weight_tensor(persistent_weights, 64, device)
    requested_steps = max(1, min(int(rollout_steps), 12))
    requested_k = max(1, min(int(top_k), 8))
    current_sparse = (
        report.get("current_sparse_code")
        if isinstance(report.get("current_sparse_code"), Mapping)
        else {}
    )
    current_indices = [
        int(value)
        for value in list(current_sparse.get("active_indices") or [])
        if isinstance(value, int) and 0 <= int(value) < 64
    ][:16]
    vocabulary_slots = [dict(slot) for slot in readout_vocabulary_slots if isinstance(slot, Mapping)][:32]
    vocabulary_count = len(vocabulary_slots)
    grounded_count = sum(1 for slot in vocabulary_slots if bool(slot.get("grounded")))
    vocabulary_candidates: list[dict[str, Any]] = []
    for slot in vocabulary_slots:
        label = _text(slot.get("label"))
        if not label:
            continue
        probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [slot],
                "device_evidence": device_report,
            }
        )
        sparse = probe.get("sparse_code_evidence") if isinstance(probe.get("sparse_code_evidence"), Mapping) else {}
        active = [
            int(value) % 64
            for value in list(sparse.get("active_indices") or [])
            if isinstance(value, int)
        ]
        folded_active = sorted({int(value) % 32 for value in active})
        vocabulary_candidates.append(
            {
                "label": label,
                "pressure_band": slot.get("pressure_band"),
                "grounded": bool(slot.get("grounded")),
                "active_indices": active[:16],
                "folded_active_indices": folded_active[:16],
            }
        )

    vector = _indices_to_vector(current_indices, 64, device)
    rollout_trace: list[dict[str, Any]] = []
    visited_hashes: set[str] = set()
    for step_index in range(requested_steps):
        logits = vector @ memory_tensor
        predicted = torch.topk(logits, k=requested_k)
        predicted_indices = [int(value) for value in predicted.indices.detach().cpu().tolist()]
        predicted_strengths = [float(value) for value in predicted.values.detach().cpu().tolist()]
        positive_strength = float(sum(max(0.0, value) for value in predicted_strengths))
        strength_by_index = {
            int(index): (
                float(predicted_strengths[position])
                if position < len(predicted_strengths)
                else 0.0
            )
            for position, index in enumerate(predicted_indices)
        }
        label_matches = []
        for candidate in vocabulary_candidates:
            matched = [index for index in candidate["active_indices"] if index in strength_by_index]
            score_by_index = strength_by_index
            if not matched:
                folded_strength_by_index = {
                    int(index) % 32: strength
                    for index, strength in strength_by_index.items()
                }
                matched = [
                    index
                    for index in candidate["folded_active_indices"]
                    if index in folded_strength_by_index
                ]
                score_by_index = folded_strength_by_index
            score = float(sum(max(0.0, score_by_index[index]) for index in matched))
            if matched and score > 0.0:
                label_matches.append(
                    {
                        "label": candidate["label"],
                        "pressure_band": candidate.get("pressure_band"),
                        "grounded": bool(candidate.get("grounded")),
                        "matched_indices": matched,
                        "score": score,
                    }
                )
        label_matches.sort(key=lambda item: (-float(item["score"]), item["label"]))
        selected = label_matches[0] if label_matches else None
        active_hash = _sha256_json(predicted_indices[:16])
        rollout_trace.append(
            {
                "step_index": step_index,
                "predicted_sparse_indices": predicted_indices,
                "predicted_sparse_strengths": predicted_strengths,
                "support_strength": positive_strength,
                "active_indices_hash": active_hash,
                "selected_label": selected.get("label") if selected else None,
                "selected_label_grounded": bool(selected.get("grounded")) if selected else False,
                "selected_label_score": float(selected.get("score")) if selected else 0.0,
                "candidate_label_count": len(label_matches),
                "candidate_labels": label_matches[:6],
            }
        )
        visited_hashes.add(active_hash)
        vector = _indices_to_vector(predicted_indices, 64, device)

    evaluation = dict(transition_memory_evaluation or {})
    evaluation_summary = (
        evaluation.get("evaluation_summary")
        if isinstance(evaluation.get("evaluation_summary"), Mapping)
        else {}
    )
    prediction_provenance = (
        report.get("provenance_evidence")
        if isinstance(report.get("provenance_evidence"), Mapping)
        else {}
    )
    evaluation_ready = bool(
        evaluation.get("surface") == "snn_language_transition_memory_prediction_evaluation.v1"
        and evaluation.get("owned_by_hecsn")
        and evaluation.get("promotion_gate", {}).get("status") == "ready_for_operator_review"
        if isinstance(evaluation.get("promotion_gate"), Mapping)
        else False
    )
    evaluation_provenance = (
        evaluation.get("provenance_evidence")
        if isinstance(evaluation.get("provenance_evidence"), Mapping)
        else {}
    )
    evaluation_hash = str(
        evaluation.get("transition_memory_evaluation_hash")
        or evaluation_provenance.get("evaluation_hash")
        or ""
    )
    evaluation_non_worsening = bool(evaluation_summary.get("worsened_sequence_count", 0) == 0)
    grounded_labels = [
        str(step.get("selected_label") or "")
        for step in rollout_trace
        if bool(step.get("selected_label_grounded")) and str(step.get("selected_label") or "")
    ]
    unique_grounded_labels = []
    for label in grounded_labels:
        if label not in unique_grounded_labels:
            unique_grounded_labels.append(label)
    rollout_hash = _sha256_json(
        {
            "prediction_hash": prediction_provenance.get("prediction_hash"),
            "persistent_transition_weights_hash": persistent_weights_hash,
            "rollout_trace": rollout_trace,
            "transition_memory_evaluation_hash": evaluation_hash or None,
        }
    )
    persistent_count = int(torch.count_nonzero(memory_tensor).item())
    grounded_fraction = grounded_count / max(1, vocabulary_count)
    ready = bool(
        report.get("owned_by_hecsn")
        and not bool(report.get("external_dependency"))
        and persistent_count > 0
        and current_indices
        and rollout_trace
        and unique_grounded_labels
        and grounded_fraction >= 0.5
        and evaluation_ready
        and evaluation_non_worsening
    )
    return {
        "artifact_kind": "terminus_snn_language_readout_rollout_candidate",
        "surface": "snn_language_readout_rollout_candidate.v1",
        "available": bool(current_indices and persistent_count > 0),
        "source": "semantics.spike_language_neurons.readout_rollout_candidate",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": True,
        "generation_scope": "bounded_grounded_readout_rollout_candidate",
        "freeform_language_generation": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "rollout": {
            "step_count": len(rollout_trace),
            "requested_steps": requested_steps,
            "top_k": requested_k,
            "labels": unique_grounded_labels[:requested_steps],
            "text": " ".join(unique_grounded_labels[:requested_steps]),
        },
        "rollout_trace": rollout_trace,
        "readout_rollout_evidence": {
            "initial_sparse_indices": current_indices,
            "visited_sparse_state_count": len(visited_hashes),
            "persistent_transition_weight_count": persistent_count,
            "persistent_transition_weights_hash": persistent_weights_hash,
            "rollout_hash": rollout_hash,
            "rollout_id": f"snn-readout-rollout:{rollout_hash[:16]}",
        },
        "grounding_evidence": {
            "vocabulary_slot_count": vocabulary_count,
            "grounded_slot_count": grounded_count,
            "grounded_fraction": float(grounded_fraction),
            "grounded_rollout_label_count": len(unique_grounded_labels),
        },
        "transition_memory_evaluation_evidence": {
            "available": bool(evaluation),
            "surface": evaluation.get("surface"),
            "owned_by_hecsn": bool(evaluation.get("owned_by_hecsn")),
            "review_ready": evaluation_ready,
            "non_worsening": evaluation_non_worsening,
            "transition_memory_evaluation_hash": evaluation_hash or None,
            "persistent_transition_weight_count": int(
                evaluation_summary.get("persistent_transition_weight_count") or 0
            ),
            "influenced_prediction_count": int(
                evaluation_summary.get("influenced_prediction_count") or 0
            ),
        },
        "provenance_evidence": {
            "rollout_hash": rollout_hash,
            "rollout_id": f"snn-readout-rollout:{rollout_hash[:16]}",
            "prediction_hash": prediction_provenance.get("prediction_hash"),
            "current_sparse_code_hash": prediction_provenance.get("current_sparse_code_hash"),
            "persistent_transition_weights_hash": persistent_weights_hash,
            "transition_memory_evaluation_hash": evaluation_hash or None,
            "hash_algorithm": "sha256_canonical_json",
        },
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "tensor_device": str(memory_tensor.device),
            "cuda_tensor": bool(memory_tensor.is_cuda),
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_readout_rollout_evidence",
            "eligible_for_bounded_readout_rollout": ready,
            "eligible_for_bounded_readout_generation": ready,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_review_snn_language_readout_rollout_candidate"
            if ready
            else "collect_readout_rollout_evidence",
            "required_evidence": {
                "prediction_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
                "external_dependency_absent": not bool(report.get("external_dependency")),
                "persistent_transition_memory_available": persistent_count > 0,
                "initial_sparse_code_available": bool(current_indices),
                "transition_memory_prediction_evaluation_ready": evaluation_ready,
                "transition_memory_prediction_non_worsening": evaluation_non_worsening,
                "grounded_vocabulary_available": grounded_fraction >= 0.5,
                "grounded_rollout_labels_available": bool(unique_grounded_labels),
                "runtime_mutation_absent": True,
            },
        },
    }


def evaluate_snn_language_readout_rollout_replay(
    readout_rollout_candidate: Mapping[str, Any],
    *,
    candidate_limit: int = 8,
    device_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only replay-review gate for a bounded readout rollout."""

    candidate = dict(readout_rollout_candidate)
    limit = max(1, min(32, int(candidate_limit)))
    rollout = candidate.get("rollout") if isinstance(candidate.get("rollout"), Mapping) else {}
    rollout_trace = [
        dict(item)
        for item in list(candidate.get("rollout_trace") or [])
        if isinstance(item, Mapping)
    ][:limit]
    rollout_evidence = (
        candidate.get("readout_rollout_evidence")
        if isinstance(candidate.get("readout_rollout_evidence"), Mapping)
        else {}
    )
    provenance = (
        candidate.get("provenance_evidence")
        if isinstance(candidate.get("provenance_evidence"), Mapping)
        else {}
    )
    transition_eval = (
        candidate.get("transition_memory_evaluation_evidence")
        if isinstance(candidate.get("transition_memory_evaluation_evidence"), Mapping)
        else {}
    )
    gate = candidate.get("promotion_gate") if isinstance(candidate.get("promotion_gate"), Mapping) else {}
    candidate_device = (
        candidate.get("device_evidence")
        if isinstance(candidate.get("device_evidence"), Mapping)
        else {}
    )
    device_report = dict(device_evidence or candidate_device)
    rollout_labels = [str(value) for value in list(rollout.get("labels") or []) if str(value)][:limit]
    replay_targets: list[dict[str, Any]] = []
    trace_hashes_valid = True
    for step in rollout_trace:
        sparse_indices = [
                int(value)
                for value in list(step.get("predicted_sparse_indices") or [])
                if isinstance(value, int) and 0 <= int(value) < 64
        ][:16]
        active_indices_hash = str(step.get("active_indices_hash") or "")
        active_hash_valid = active_indices_hash == _sha256_json(sparse_indices)
        trace_hashes_valid = trace_hashes_valid and active_hash_valid
        label = str(step.get("selected_label") or "")
        grounded = bool(step.get("selected_label_grounded"))
        if not label or not grounded:
            continue
        replay_targets.append(
            {
                "step_index": int(step.get("step_index") or len(replay_targets)),
                "selected_label": label,
                "grounded": True,
                "selection_score": float(step.get("selected_label_score") or 0.0),
                "transition_support": float(step.get("support_strength") or 0.0),
                "predicted_sparse_indices": sparse_indices,
                "active_indices_hash": active_indices_hash,
                "active_indices_hash_valid": active_hash_valid,
            }
        )
    rollout_hash = str(provenance.get("rollout_hash") or rollout_evidence.get("rollout_hash") or "")
    required_provenance = {
        "rollout_hash": rollout_hash,
        "rollout_id": provenance.get("rollout_id"),
        "prediction_hash": provenance.get("prediction_hash"),
        "current_sparse_code_hash": provenance.get("current_sparse_code_hash"),
        "persistent_transition_weights_hash": provenance.get("persistent_transition_weights_hash"),
        "transition_memory_evaluation_hash": provenance.get("transition_memory_evaluation_hash"),
    }
    evaluation_material = {
        "surface": "snn_language_readout_rollout_replay_evaluation.v1",
        "candidate_limit": limit,
        "rollout_hash": rollout_hash,
        "rollout_id": required_provenance["rollout_id"],
        "replay_targets": replay_targets,
    }
    evaluation_hash = _sha256_json(evaluation_material)
    required = {
        "candidate_surface_available": candidate.get("surface") == "snn_language_readout_rollout_candidate.v1",
        "candidate_owned_by_hecsn": bool(candidate.get("owned_by_hecsn")),
        "external_dependency_absent": not bool(candidate.get("external_dependency")),
        "external_checkpoint_absent": not bool(candidate.get("loads_external_checkpoint")),
        "candidate_gate_ready": bool(gate.get("eligible_for_bounded_readout_rollout")),
        "runtime_mutation_absent": not bool(candidate.get("mutates_runtime_state")),
        "training_absent": not bool(candidate.get("trains_runtime_model")),
        "plasticity_absent": not bool(candidate.get("applies_plasticity")),
        "freeform_generation_absent": not bool(candidate.get("freeform_language_generation")),
        "provenance_complete": all(bool(value) for value in required_provenance.values()),
        "transition_memory_review_ready": bool(transition_eval.get("review_ready")),
        "transition_memory_non_worsening": bool(transition_eval.get("non_worsening")),
        "grounded_rollout_labels_available": bool(rollout_labels) and bool(replay_targets),
        "replay_targets_grounded": bool(replay_targets) and all(target["grounded"] for target in replay_targets),
        "trace_hashes_valid": bool(rollout_trace) and trace_hashes_valid,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_readout_rollout_replay_evaluation",
        "surface": "snn_language_readout_rollout_replay_evaluation.v1",
        "available": bool(candidate),
        "source": "semantics.spike_language_neurons.readout_rollout_replay_evaluation",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "freeform_language_generation": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "recorded_in_ledger": False,
        "eligible_for_replay_priority": False,
        "eligible_for_replay_memory": False,
        "replay_evaluation": {
            "candidate_limit": limit,
            "trace_step_count": len(rollout_trace),
            "rollout_label_count": len(rollout_labels),
            "target_count": len(replay_targets),
            "trace_hashes_valid": trace_hashes_valid,
            "replay_targets": replay_targets,
        },
        "provenance_evidence": {
            "rollout_replay_evaluation_hash": evaluation_hash,
            "rollout_replay_evaluation_id": f"snn-readout-rollout-replay-eval:{evaluation_hash[:16]}",
            "rollout_hash": rollout_hash or None,
            "rollout_id": required_provenance["rollout_id"],
            "prediction_hash": required_provenance["prediction_hash"],
            "current_sparse_code_hash": required_provenance["current_sparse_code_hash"],
            "persistent_transition_weights_hash": required_provenance["persistent_transition_weights_hash"],
            "transition_memory_evaluation_hash": required_provenance["transition_memory_evaluation_hash"],
            "hash_algorithm": "sha256_canonical_json",
        },
        "device_evidence": {
            "requested_device": device_report.get("device")
            or device_report.get("requested_device")
            or candidate_device.get("requested_device"),
            "tensor_device": candidate_device.get("tensor_device"),
            "cuda_tensor": bool(candidate_device.get("cuda_tensor")),
            "device_source": device_report.get("source")
            or device_report.get("device_source")
            or candidate_device.get("device_source"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_readout_rollout_replay_evidence",
            "eligible_for_readout_rollout_ledger_recording_review": ready,
            "eligible_for_replay_priority": False,
            "eligible_for_replay_memory": False,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "eligible_for_plasticity_application": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_review_record_snn_language_readout_rollout_evidence"
            if ready
            else "collect_grounded_readout_rollout_replay_evidence",
            "required_evidence": required,
        },
    }


def build_snn_language_transition_memory_prediction_evaluation(
    training_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    evaluation_readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    transition_memory_state: Mapping[str, Any],
    device_evidence: Mapping[str, Any] | None = None,
    *,
    learning_rate: float = 0.08,
    epochs: int = 2,
    top_k: int = 8,
) -> dict[str, Any]:
    """Compare baseline vs persistent-memory sparse next-code prediction."""

    state = dict(transition_memory_state or {})
    persistent_weights = (
        state.get("sparse_transition_weights")
        if isinstance(state.get("sparse_transition_weights"), Mapping)
        else {}
    )
    transition_memory_hash = _sha256_json(dict(persistent_weights))
    training_sequence_hash = _sha256_json(
        [[dict(slot) for slot in batch if isinstance(slot, Mapping)] for batch in training_readout_slot_batches]
    )
    pairs: list[tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]] = []
    normalized_eval = [
        [dict(slot) for slot in batch if isinstance(slot, Mapping)]
        for batch in evaluation_readout_slot_batches
    ]
    evaluation_window_hash = _sha256_json(normalized_eval)
    for index in range(max(0, len(normalized_eval) - 1)):
        current = normalized_eval[index]
        observed = normalized_eval[index + 1]
        if current and observed:
            pairs.append((current, observed))
    records: list[dict[str, Any]] = []
    baseline_scores: list[float] = []
    memory_scores: list[float] = []
    influenced_count = 0
    improved_count = 0
    worsened_count = 0
    for pair_index, (current_slots, observed_slots) in enumerate(pairs):
        baseline_prediction = predict_spike_language_sequence(
            training_readout_slot_batches,
            current_slots,
            device_evidence,
            learning_rate=learning_rate,
            epochs=epochs,
            top_k=top_k,
            persistent_transition_weights={},
        )
        memory_prediction = predict_spike_language_sequence(
            training_readout_slot_batches,
            current_slots,
            device_evidence,
            learning_rate=learning_rate,
            epochs=epochs,
            top_k=top_k,
            persistent_transition_weights=persistent_weights,
        )
        baseline_mismatch = evaluate_spike_language_sequence_mismatch(
            baseline_prediction,
            observed_slots,
            device_evidence,
        )
        memory_mismatch = evaluate_spike_language_sequence_mismatch(
            memory_prediction,
            observed_slots,
            device_evidence,
        )
        baseline_score = _float(
            (baseline_mismatch.get("prediction_error") or {}).get("mismatch_score")
            if isinstance(baseline_mismatch.get("prediction_error"), Mapping)
            else None,
            1.0,
        )
        memory_score = _float(
            (memory_mismatch.get("prediction_error") or {}).get("mismatch_score")
            if isinstance(memory_mismatch.get("prediction_error"), Mapping)
            else None,
            1.0,
        )
        delta = baseline_score - memory_score
        baseline_scores.append(baseline_score)
        memory_scores.append(memory_score)
        persistent_evidence = (
            memory_prediction.get("persistent_transition_evidence")
            if isinstance(memory_prediction.get("persistent_transition_evidence"), Mapping)
            else {}
        )
        baseline_provenance = (
            baseline_prediction.get("provenance_evidence")
            if isinstance(baseline_prediction.get("provenance_evidence"), Mapping)
            else {}
        )
        memory_provenance = (
            memory_prediction.get("provenance_evidence")
            if isinstance(memory_prediction.get("provenance_evidence"), Mapping)
            else {}
        )
        current_sparse = (
            memory_prediction.get("current_sparse_code")
            if isinstance(memory_prediction.get("current_sparse_code"), Mapping)
            else {}
        )
        current_sparse_hash = str(current_sparse.get("active_indices_hash") or "")
        observed_sparse = (
            memory_mismatch.get("observed_sparse_code")
            if isinstance(memory_mismatch.get("observed_sparse_code"), Mapping)
            else {}
        )
        influenced = bool(persistent_evidence.get("influenced_prediction"))
        influenced_count += int(influenced)
        improved_count += int(delta > 1e-9)
        worsened_count += int(delta < -1e-9)
        records.append(
            {
                "pair_index": pair_index,
                "baseline_mismatch_score": float(baseline_score),
                "memory_mismatch_score": float(memory_score),
                "mismatch_delta": float(delta),
                "persistent_memory_influenced_prediction": influenced,
                "current_sparse_code_hash": current_sparse_hash or None,
                "current_readout_hash": memory_provenance.get("current_readout_hash"),
                "baseline_prediction_hash": baseline_provenance.get("prediction_hash"),
                "memory_prediction_hash": memory_provenance.get("prediction_hash"),
                "observed_sparse_code_hash": _sha256_json(
                    [
                        int(value)
                        for value in list(observed_sparse.get("active_indices") or [])
                        if isinstance(value, int)
                    ][:16]
                ),
                "observed_slot_count": len(observed_slots),
            }
        )
    baseline_mean = sum(baseline_scores) / max(1, len(baseline_scores))
    memory_mean = sum(memory_scores) / max(1, len(memory_scores))
    mean_delta = baseline_mean - memory_mean
    persistent_count = len(dict(persistent_weights))
    non_worse = bool(records) and mean_delta >= -1e-9 and worsened_count == 0
    ready = persistent_count > 0 and bool(records) and non_worse
    evaluated_current_sparse_hashes = sorted(
        {
            str(record.get("current_sparse_code_hash") or "")
            for record in records
            if str(record.get("current_sparse_code_hash") or "")
        }
    )[:32]
    evaluated_current_readout_hashes = sorted(
        {
            str(record.get("current_readout_hash") or "")
            for record in records
            if str(record.get("current_readout_hash") or "")
        }
    )[:32]
    evaluated_prediction_hashes = sorted(
        {
            str(record.get("memory_prediction_hash") or "")
            for record in records
            if str(record.get("memory_prediction_hash") or "")
        }
    )[:32]
    evaluation_hash = _sha256_json(
        {
            "training_window_hash": training_sequence_hash,
            "evaluation_window_hash": evaluation_window_hash,
            "persistent_transition_weights_hash": transition_memory_hash,
            "evaluation_summary": {
                "evaluation_pair_count": len(records),
                "baseline_mean_mismatch_score": float(baseline_mean),
                "memory_mean_mismatch_score": float(memory_mean),
                "mean_mismatch_delta": float(mean_delta),
                "influenced_prediction_count": int(influenced_count),
                "improved_sequence_count": int(improved_count),
                "worsened_sequence_count": int(worsened_count),
                "persistent_transition_weight_count": int(persistent_count),
            },
            "evaluated_prediction_hashes": evaluated_prediction_hashes,
            "sequence_records": records[:16],
        }
    )
    return {
        "artifact_kind": "terminus_snn_language_transition_memory_prediction_evaluation",
        "surface": "snn_language_transition_memory_prediction_evaluation.v1",
        "available": bool(records),
        "source": "semantics.spike_language_neurons.transition_memory_prediction_evaluation",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "evaluation_summary": {
            "evaluation_pair_count": len(records),
            "baseline_mean_mismatch_score": float(baseline_mean),
            "memory_mean_mismatch_score": float(memory_mean),
            "mean_mismatch_delta": float(mean_delta),
            "influenced_prediction_count": int(influenced_count),
            "improved_sequence_count": int(improved_count),
            "worsened_sequence_count": int(worsened_count),
            "persistent_transition_weight_count": int(persistent_count),
            "evaluated_current_sparse_hashes": evaluated_current_sparse_hashes,
            "evaluated_current_readout_hashes": evaluated_current_readout_hashes,
            "evaluated_prediction_hashes": evaluated_prediction_hashes,
        },
        "provenance_evidence": {
            "evaluation_hash": evaluation_hash,
            "evaluation_id": f"snn-transition-eval:{evaluation_hash[:16]}",
            "training_sequence_hash": training_sequence_hash,
            "training_window_hash": training_sequence_hash,
            "evaluation_window_hash": evaluation_window_hash,
            "transition_memory_hash": transition_memory_hash,
            "persistent_transition_weights_hash": transition_memory_hash,
            "evaluated_current_sparse_hashes": evaluated_current_sparse_hashes,
            "evaluated_current_readout_hashes": evaluated_current_readout_hashes,
            "evaluated_prediction_hashes": evaluated_prediction_hashes,
            "hash_algorithm": "sha256_canonical_json",
        },
        "sequence_records": records[:16],
        "promotion_gate": {
            "status": "ready_for_operator_review"
            if ready
            else "blocked_missing_transition_memory_evidence"
            if persistent_count <= 0
            else "collect_non_worsening_prediction_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_bounded_readout_generation_review": ready,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "eligible_for_runtime_training": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_review_snn_language_readout_draft"
            if ready
            else "collect_grounded_transition_memory_evaluation_window",
            "required_evidence": {
                "persistent_transition_memory_available": persistent_count > 0,
                "evaluation_windows_available": bool(records),
                "persistent_memory_non_worsening": non_worse,
                "external_dependency_absent": True,
                "runtime_mutation_absent": True,
            },
        },
    }


def evaluate_spike_language_sequence_mismatch(
    prediction_report: Mapping[str, Any],
    observed_readout_slots: Sequence[Mapping[str, Any]],
    device_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare predicted sparse code with observed sparse code without learning."""

    report = dict(prediction_report)
    prediction = report.get("prediction") if isinstance(report.get("prediction"), Mapping) else {}
    device_report = dict(device_evidence or report.get("device_evidence") or {})
    observed_probe = build_spike_language_decoder_probe(
        {
            "readout_slots": [dict(slot) for slot in observed_readout_slots if isinstance(slot, Mapping)],
            "device_evidence": device_report,
        }
    )
    observed_sparse = (
        observed_probe.get("sparse_code_evidence")
        if isinstance(observed_probe.get("sparse_code_evidence"), Mapping)
        else {}
    )
    predicted_indices = {
        int(value) % 64
        for value in list(prediction.get("predicted_sparse_indices") or [])
        if isinstance(value, int)
    }
    observed_indices = {
        int(value) % 64
        for value in list(observed_sparse.get("active_indices") or [])
        if isinstance(value, int)
    }
    matched = predicted_indices.intersection(observed_indices)
    union = predicted_indices.union(observed_indices)
    precision = len(matched) / max(1, len(predicted_indices))
    recall = len(matched) / max(1, len(observed_indices))
    mismatch = 1.0 - (len(matched) / max(1, len(union)))
    prediction_available = bool(report.get("available")) and bool(predicted_indices)
    observed_available = bool(observed_indices)
    ready = prediction_available and observed_available
    return {
        "artifact_kind": "terminus_snn_language_sequence_mismatch_probe",
        "surface": "snn_language_sequence_mismatch_probe.v1",
        "available": ready,
        "source": "semantics.spike_language_neurons.sequence_mismatch_probe",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "prediction_surface": report.get("surface"),
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "tensor_device": (observed_probe.get("device_evidence") or {}).get("tensor_device")
            if isinstance(observed_probe.get("device_evidence"), Mapping)
            else None,
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "prediction_error": {
            "predicted_index_count": len(predicted_indices),
            "observed_index_count": len(observed_indices),
            "matched_index_count": len(matched),
            "precision": float(precision),
            "recall": float(recall),
            "mismatch_score": float(mismatch),
            "prediction_error_band": _mismatch_band(mismatch),
        },
        "sparse_code_delta": {
            "predicted_only_indices": sorted(predicted_indices.difference(observed_indices))[:16],
            "observed_only_indices": sorted(observed_indices.difference(predicted_indices))[:16],
            "matched_indices": sorted(matched)[:16],
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "collect_prediction_and_observation",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_learning_signal": False,
            "requires_operator_approval": ready,
            "next_gate": "operator_approved_snn_language_prediction_error_evaluation"
            if ready
            else "collect_observed_sparse_code",
        },
    }


def build_spike_language_plasticity_pressure(
    mismatch_report: Mapping[str, Any],
    *,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert sparse prediction error into read-only local plasticity pressure."""

    report = dict(mismatch_report)
    error = report.get("prediction_error") if isinstance(report.get("prediction_error"), Mapping) else {}
    delta = report.get("sparse_code_delta") if isinstance(report.get("sparse_code_delta"), Mapping) else {}
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    mismatch_score = _float(error.get("mismatch_score"), 1.0)
    observed_only = [int(value) for value in list(delta.get("observed_only_indices") or []) if isinstance(value, int)]
    predicted_only = [int(value) for value in list(delta.get("predicted_only_indices") or []) if isinstance(value, int)]
    matched = [int(value) for value in list(delta.get("matched_indices") or []) if isinstance(value, int)]
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    pressure_score = min(1.0, max(0.0, mismatch_score))
    if pressure_score >= 0.66:
        pressure_band = "high"
        update_focus = "review_local_sequence_transition_growth"
    elif pressure_score >= 0.25:
        pressure_band = "medium"
        update_focus = "review_local_sequence_weight_rebalance"
    else:
        pressure_band = "low"
        update_focus = "monitor_sequence_prediction"
    required = {
        "mismatch_available": bool(report.get("available")),
        "mismatch_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "external_dependency_absent": not bool(report.get("external_dependency")),
        "generation_absent": not bool(report.get("generates_text")),
        "runtime_training_absent": not bool(report.get("trains_runtime_model")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "mismatch_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "prediction_error_measured": "mismatch_score" in error,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values()) and pressure_band in {"medium", "high"}
    return {
        "artifact_kind": "terminus_snn_language_plasticity_pressure_gate",
        "surface": "snn_language_plasticity_pressure.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_pressure",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "mismatch_surface": report.get("surface"),
        "plasticity_pressure": {
            "pressure_score": float(pressure_score),
            "pressure_band": pressure_band,
            "update_focus": update_focus,
            "observed_only_index_count": len(observed_only),
            "predicted_only_index_count": len(predicted_only),
            "matched_index_count": len(matched),
        },
        "candidate_update": {
            "target": "local_snn_language_sequence_transition_weights",
            "rule_family": "error_modulated_local_hebbian_sequence_update",
            "increase_support_for_indices": observed_only[:16],
            "decrease_support_for_indices": predicted_only[:16],
            "preserve_support_for_indices": matched[:16],
            "requires_isolated_replay": True,
            "requires_operator_approval": True,
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "monitor_or_collect_more_mismatch_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_learning_signal": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_plasticity_design_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_isolated_language_plasticity_trial"
            if ready
            else "collect_prediction_error_window",
            "required_evidence": required,
        },
        "success_evidence": [
            "sequence_mismatch_report",
            "prediction_error_window",
            "local_plasticity_rule_design",
            "runtime_truth_delta",
            "rollback_policy",
            "device_evidence_report",
        ],
    }


def run_spike_language_plasticity_trial(
    pressure_report: Mapping[str, Any],
    *,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a local plasticity update from pressure evidence without applying it."""

    report = dict(pressure_report)
    pressure = report.get("plasticity_pressure") if isinstance(report.get("plasticity_pressure"), Mapping) else {}
    candidate = report.get("candidate_update") if isinstance(report.get("candidate_update"), Mapping) else {}
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    pressure_score = _float(pressure.get("pressure_score"), 0.0)
    increase_indices = [
        int(value)
        for value in list(candidate.get("increase_support_for_indices") or [])
        if isinstance(value, int)
    ]
    decrease_indices = [
        int(value)
        for value in list(candidate.get("decrease_support_for_indices") or [])
        if isinstance(value, int)
    ]
    preserve_indices = [
        int(value)
        for value in list(candidate.get("preserve_support_for_indices") or [])
        if isinstance(value, int)
    ]
    touched_count = len(set(increase_indices + decrease_indices + preserve_indices))
    correction_capacity = min(1.0, (len(increase_indices) + len(decrease_indices)) / max(1, touched_count or 1))
    expected_reduction = min(pressure_score, pressure_score * 0.5 * correction_capacity)
    post_pressure_score = max(0.0, pressure_score - expected_reduction)
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    required = {
        "pressure_available": bool(report.get("available")),
        "pressure_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "pressure_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "plasticity_application_absent": not bool(report.get("applies_plasticity")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "candidate_indices_available": touched_count > 0,
        "expected_reduction_positive": expected_reduction > 0.0,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_trial",
        "surface": "snn_language_plasticity_trial.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_trial",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "returns_trained_weights": False,
        "mutates_runtime_state": False,
        "pressure_surface": report.get("surface"),
        "trial_summary": {
            "pre_pressure_score": float(pressure_score),
            "expected_pressure_reduction": float(expected_reduction),
            "post_pressure_score": float(post_pressure_score),
            "pre_pressure_band": _mismatch_band(pressure_score),
            "post_pressure_band": _mismatch_band(post_pressure_score),
            "candidate_index_count": touched_count,
        },
        "ephemeral_update": {
            "rule_family": "error_modulated_local_hebbian_sequence_update",
            "increase_support_for_indices": increase_indices[:16],
            "decrease_support_for_indices": decrease_indices[:16],
            "preserve_support_for_indices": preserve_indices[:16],
            "weights_persisted": False,
            "runtime_update_applied": False,
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_plasticity_trial_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_isolated_replay_evaluation": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_isolated_language_plasticity_replay"
            if ready
            else "collect_plasticity_trial_evidence",
            "required_evidence": required,
        },
    }


def evaluate_spike_language_plasticity_replay(
    trial_report: Mapping[str, Any],
    *,
    replay_window: Sequence[Mapping[str, Any]] | None = None,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a plasticity trial is ready for isolated replay review."""

    report = dict(trial_report)
    summary = report.get("trial_summary") if isinstance(report.get("trial_summary"), Mapping) else {}
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    replay_items = [dict(item) for item in list(replay_window or []) if isinstance(item, Mapping)]
    pre_pressure = _float(summary.get("pre_pressure_score"), 1.0)
    post_pressure = _float(summary.get("post_pressure_score"), pre_pressure)
    reduction = _float(summary.get("expected_pressure_reduction"), 0.0)
    replay_window_available = len(replay_items) > 0
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    non_worsening = post_pressure <= pre_pressure
    required = {
        "trial_available": bool(report.get("available")),
        "trial_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "trial_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "plasticity_application_absent": not bool(report.get("applies_plasticity")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "weights_absent": not bool(report.get("returns_trained_weights")),
        "expected_reduction_positive": reduction > 0.0,
        "pressure_non_worsening": non_worsening,
        "replay_window_available": replay_window_available,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_replay_evaluation",
        "surface": "snn_language_plasticity_replay_evaluation.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_replay_evaluation",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "trial_surface": report.get("surface"),
        "replay_evidence": {
            "replay_window_count": len(replay_items),
            "pre_pressure_score": float(pre_pressure),
            "post_pressure_score": float(post_pressure),
            "expected_pressure_reduction": float(reduction),
            "pressure_non_worsening": bool(non_worsening),
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_replay_evaluation_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_replay_promotion": False,
            "eligible_for_operator_replay_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_language_plasticity_replay_experiment"
            if ready
            else "collect_replay_evaluation_window",
            "required_evidence": required,
        },
    }


def run_spike_language_plasticity_replay_experiment(
    replay_evaluation: Mapping[str, Any],
    *,
    replay_sequences: Sequence[Mapping[str, Any]] | None = None,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an isolated sparse-code replay experiment without applying plasticity."""

    report = dict(replay_evaluation)
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    evidence = report.get("replay_evidence") if isinstance(report.get("replay_evidence"), Mapping) else {}
    sequences = [dict(item) for item in list(replay_sequences or []) if isinstance(item, Mapping)]
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    replay_count = len(sequences)
    grounded_count = sum(1 for item in sequences if bool(item.get("grounded", True)))
    coverage = grounded_count / replay_count if replay_count else 0.0
    pre_pressure = _float(evidence.get("pre_pressure_score"), 1.0)
    post_pressure = _float(evidence.get("post_pressure_score"), pre_pressure)
    expected_reduction = _float(evidence.get("expected_pressure_reduction"), 0.0)
    replay_gain = expected_reduction * coverage
    simulated_post_pressure = max(0.0, post_pressure - replay_gain)
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    pressure_stable = simulated_post_pressure <= pre_pressure
    trace = [
        {
            "sequence_id": _text(item.get("sequence_id") or item.get("case_id") or index),
            "grounded": bool(item.get("grounded", True)),
            "applied_to_runtime": False,
            "weights_persisted": False,
        }
        for index, item in enumerate(sequences)
    ]
    required = {
        "replay_evaluation_available": bool(report.get("available")),
        "replay_evaluation_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "replay_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "plasticity_application_absent": not bool(report.get("applies_plasticity")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "replay_sequences_available": replay_count > 0,
        "grounded_replay_coverage_sufficient": coverage >= 0.5,
        "pressure_stable_after_replay": pressure_stable,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_replay_experiment",
        "surface": "snn_language_plasticity_replay_experiment.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_replay_experiment",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "replay_evaluation_surface": report.get("surface"),
        "replay_experiment": {
            "replay_sequence_count": replay_count,
            "grounded_replay_sequence_count": grounded_count,
            "grounded_replay_coverage": float(coverage),
            "pre_pressure_score": float(pre_pressure),
            "post_evaluation_pressure_score": float(post_pressure),
            "simulated_post_replay_pressure_score": float(simulated_post_pressure),
            "expected_replay_pressure_gain": float(replay_gain),
            "pressure_stable_after_replay": bool(pressure_stable),
        },
        "ephemeral_replay": {
            "trace": trace,
            "weights_persisted": False,
            "runtime_update_applied": False,
            "runtime_state_mutated": False,
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_replay_experiment_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_replay_promotion": False,
            "eligible_for_operator_application_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_language_plasticity_application_design"
            if ready
            else "collect_isolated_replay_experiment_evidence",
            "required_evidence": required,
        },
    }


def build_spike_language_plasticity_application_design(
    replay_experiment: Mapping[str, Any],
    *,
    application_policy: Mapping[str, Any] | None = None,
    device_evidence: Mapping[str, Any] | None = None,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Design a bounded plasticity application plan without applying it."""

    report = dict(replay_experiment)
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    experiment = report.get("replay_experiment") if isinstance(report.get("replay_experiment"), Mapping) else {}
    policy = dict(application_policy or {})
    device_report = dict(device_evidence or policy.get("device_evidence") or report.get("device_evidence") or {})
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    learning_rate = min(max(_float(policy.get("learning_rate"), 0.02), 0.0), 0.25)
    max_weight_delta = min(max(_float(policy.get("max_weight_delta"), 0.05), 0.0), 0.25)
    locality_radius = max(int(_float(policy.get("locality_radius"), 1.0)), 1)
    coverage = _float(experiment.get("grounded_replay_coverage"), 0.0)
    pressure_stable = bool(experiment.get("pressure_stable_after_replay"))
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    normalization_enabled = bool(policy.get("normalization", True))
    local_only = bool(policy.get("local_only", True))
    selected_device = _text(device_report.get("device") or device_report.get("tensor_device") or "unknown")
    device_report_available = bool(device_report) and bool(device_report.get("device_report_available", True))
    required = {
        "replay_experiment_available": bool(report.get("available")),
        "replay_experiment_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "replay_experiment_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "plasticity_application_absent": not bool(report.get("applies_plasticity")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "weights_absent": not bool(report.get("returns_trained_weights")),
        "grounded_replay_coverage_sufficient": coverage >= 0.5,
        "pressure_stable_after_replay": pressure_stable,
        "learning_rate_bounded": 0.0 < learning_rate <= 0.25,
        "max_weight_delta_bounded": 0.0 < max_weight_delta <= 0.25,
        "locality_radius_bounded": 1 <= locality_radius <= 8,
        "normalization_enabled": normalization_enabled,
        "local_update_only": local_only,
        "device_evidence_available": device_report_available,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_application_design",
        "surface": "snn_language_plasticity_application_design.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_application_design",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "replay_experiment_surface": report.get("surface"),
        "device_evidence": {
            "requested_device": selected_device,
            "tensor_device": selected_device,
            "cuda_tensor": selected_device.startswith("cuda"),
            "device_source": device_report.get("source") or device_report.get("device_source"),
            "device_report_available": device_report_available,
        },
        "application_design": {
            "learning_rate": float(learning_rate),
            "max_weight_delta": float(max_weight_delta),
            "locality_radius": locality_radius,
            "normalization": normalization_enabled,
            "local_only": local_only,
            "grounded_replay_coverage": float(coverage),
            "pressure_stable_after_replay": pressure_stable,
            "runtime_update_applied": False,
            "weights_persisted": False,
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_application_design_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_live_application": False,
            "eligible_for_operator_application_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_bounded_language_plasticity_application"
            if ready
            else "collect_application_design_evidence",
            "required_evidence": required,
        },
    }


def evaluate_spike_language_plasticity_shadow_application(
    application_design: Mapping[str, Any],
    *,
    shadow_delta: Mapping[str, Any] | None = None,
    device_evidence: Mapping[str, Any] | None = None,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a shadow plasticity application without mutating runtime state."""

    report = dict(application_design)
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    design = report.get("application_design") if isinstance(report.get("application_design"), Mapping) else {}
    delta = dict(shadow_delta or {})
    device_report = dict(device_evidence or delta.get("device_evidence") or report.get("device_evidence") or {})
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    max_allowed_delta = _float(design.get("max_weight_delta"), 0.0)
    observed_delta = abs(_float(delta.get("max_abs_weight_delta"), 0.0))
    affected_synapses = max(int(_float(delta.get("affected_synapse_count"), 0.0)), 0)
    locality_radius = max(int(_float(delta.get("locality_radius"), design.get("locality_radius"))), 0)
    designed_radius = max(int(_float(design.get("locality_radius"), 0.0)), 0)
    pressure_before = _float(delta.get("pressure_before"), 1.0)
    pressure_after = _float(delta.get("pressure_after"), pressure_before)
    selected_device = _text(device_report.get("device") or device_report.get("tensor_device") or "unknown")
    device_report_available = bool(device_report) and bool(device_report.get("device_report_available", True))
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    non_worsening = pressure_after <= pressure_before
    required = {
        "application_design_available": bool(report.get("available")),
        "application_design_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "application_design_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "design_did_not_apply_plasticity": not bool(report.get("applies_plasticity")),
        "design_did_not_mutate_runtime": not bool(report.get("mutates_runtime_state")),
        "shadow_delta_available": bool(delta),
        "shadow_delta_within_weight_bound": 0.0 <= observed_delta <= max_allowed_delta,
        "shadow_delta_has_local_support": affected_synapses > 0,
        "shadow_delta_within_locality": designed_radius > 0 and locality_radius <= designed_radius,
        "shadow_pressure_non_worsening": non_worsening,
        "device_evidence_available": device_report_available,
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_shadow_application",
        "surface": "snn_language_plasticity_shadow_application.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_shadow_application",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "application_design_surface": report.get("surface"),
        "device_evidence": {
            "requested_device": selected_device,
            "tensor_device": selected_device,
            "cuda_tensor": selected_device.startswith("cuda"),
            "device_source": device_report.get("source") or device_report.get("device_source"),
            "device_report_available": device_report_available,
        },
        "shadow_application": {
            "max_allowed_weight_delta": float(max_allowed_delta),
            "observed_max_abs_weight_delta": float(observed_delta),
            "affected_synapse_count": affected_synapses,
            "designed_locality_radius": designed_radius,
            "observed_locality_radius": locality_radius,
            "pressure_before": float(pressure_before),
            "pressure_after": float(pressure_after),
            "pressure_non_worsening": bool(non_worsening),
            "runtime_update_applied": False,
            "weights_persisted": False,
        },
        "runtime_truth_delta": truth_delta,
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_shadow_application_evidence",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_live_application": False,
            "eligible_for_operator_live_application_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_live_language_plasticity_application"
            if ready
            else "collect_shadow_application_evidence",
            "required_evidence": required,
        },
    }


def build_spike_language_plasticity_shadow_delta(
    application_design: Mapping[str, Any],
    replay_sequences: Sequence[Mapping[str, Any]],
    *,
    device_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a bounded local shadow delta from sparse replay evidence."""

    report = dict(application_design)
    design = report.get("application_design") if isinstance(report.get("application_design"), Mapping) else {}
    device_report = dict(device_evidence or report.get("device_evidence") or {})
    selected_device = _text(device_report.get("device") or device_report.get("tensor_device") or "cpu")
    device = _safe_tensor_device(selected_device)
    max_weight_delta = _float(design.get("max_weight_delta"), 0.0)
    learning_rate = _float(design.get("learning_rate"), 0.0)
    locality_radius = max(int(_float(design.get("locality_radius"), 1.0)), 1)
    rows: list[int] = []
    cols: list[int] = []
    provenance_by_synapse: dict[tuple[int, int], dict[str, Any]] = {}
    for item in replay_sequences:
        if not isinstance(item, Mapping):
            continue
        pre_indices = [int(value) for value in list(item.get("pre_indices") or []) if isinstance(value, int)]
        post_indices = [int(value) for value in list(item.get("post_indices") or []) if isinstance(value, int)]
        if not pre_indices:
            pre_indices = [int(value) for value in list(item.get("active_indices") or []) if isinstance(value, int)]
        if not post_indices:
            post_indices = [int(value) for value in list(item.get("target_indices") or []) if isinstance(value, int)]
        for pre in pre_indices:
            for post in post_indices:
                if abs(int(post) - int(pre)) <= locality_radius:
                    rows.append(int(pre))
                    cols.append(int(post))
                    provenance_by_synapse.setdefault(
                        (int(pre), int(post)),
                        {
                            "sequence_id": item.get("sequence_id"),
                            "grounded": bool(item.get("grounded", True)),
                            "readout_evidence_hash": item.get("readout_evidence_hash"),
                            "prediction_hash": item.get("prediction_hash"),
                            "transition_memory_evaluation_hash": item.get(
                                "transition_memory_evaluation_hash"
                            ),
                            "persistent_transition_weights_hash": item.get(
                                "persistent_transition_weights_hash"
                            ),
                            "source_pre_indices": pre_indices,
                            "source_post_indices": post_indices,
                            "source_active_indices": [
                                int(value)
                                for value in list(item.get("active_indices") or [])
                                if isinstance(value, int)
                            ],
                        },
                    )
    affected = len(set(zip(rows, cols)))
    bounded_synapses = [
        {
            "pre_index": int(pre),
            "post_index": int(post),
            **{
                key: value
                for key, value in provenance_by_synapse.get((int(pre), int(post)), {}).items()
                if value is not None
            },
        }
        for pre, post in sorted(set(zip(rows, cols)))
    ]
    observed_delta = min(max_weight_delta, learning_rate) if affected > 0 else 0.0
    pressure_before = _float(design.get("grounded_replay_coverage"), 0.0)
    pressure_after = max(0.0, pressure_before - observed_delta)
    delta_tensor = torch.tensor([observed_delta], device=device)
    return {
        "artifact_kind": "terminus_snn_language_plasticity_shadow_delta",
        "surface": "snn_language_plasticity_shadow_delta.v1",
        "available": affected > 0,
        "source": "semantics.spike_language_neurons.plasticity_shadow_delta",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "max_abs_weight_delta": float(delta_tensor.item()),
        "affected_synapse_count": affected,
        "bounded_synapses": bounded_synapses,
        "locality_radius": locality_radius,
        "pressure_before": float(pressure_before),
        "pressure_after": float(pressure_after),
        "device_evidence": {
            "requested_device": selected_device,
            "tensor_device": str(delta_tensor.device),
            "cuda_tensor": bool(delta_tensor.is_cuda),
            "device_source": device_report.get("source") or device_report.get("device_source"),
            "device_report_available": bool(device_report) and bool(device_report.get("device_report_available", True)),
        },
    }


def evaluate_spike_language_plasticity_live_application_readiness(
    shadow_application: Mapping[str, Any],
    *,
    rollback_readiness: Mapping[str, Any] | None = None,
    operator_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether live language plasticity could be reviewed without applying it."""

    report = dict(shadow_application)
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    rollback = dict(rollback_readiness or {})
    approval = dict(operator_approval or {})
    checkpoint_available = bool(rollback.get("checkpoint_available") or rollback.get("available"))
    restore_available = bool(rollback.get("restore_endpoint_available") or rollback.get("reversible"))
    approval_available = bool(approval.get("approved") or approval.get("operator_approved"))
    required = {
        "shadow_application_available": bool(report.get("available")),
        "shadow_application_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "shadow_application_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "shadow_did_not_apply_plasticity": not bool(report.get("applies_plasticity")),
        "shadow_did_not_mutate_runtime": not bool(report.get("mutates_runtime_state")),
        "checkpoint_available": checkpoint_available,
        "restore_endpoint_available": restore_available,
        "operator_approval_available": approval_available,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_live_application_readiness",
        "surface": "snn_language_plasticity_live_application_readiness.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_live_application_readiness",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "shadow_application_surface": report.get("surface"),
        "rollback_readiness": {
            "checkpoint_available": checkpoint_available,
            "checkpoint_path": rollback.get("checkpoint_path"),
            "restore_endpoint_available": restore_available,
            "rollback_policy_required": True,
        },
        "operator_approval": {
            "approved": approval_available,
            "operator_id": approval.get("operator_id"),
            "approval_id": approval.get("approval_id"),
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_live_application_readiness",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_live_application": False,
            "eligible_for_operator_live_application_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_approved_checkpoint_backed_live_language_plasticity_application"
            if ready
            else "collect_checkpoint_rollback_and_operator_approval",
            "required_evidence": required,
        },
    }


def evaluate_spike_language_plasticity_live_application_preflight(
    live_application_readiness: Mapping[str, Any],
    *,
    application_target: Mapping[str, Any] | None = None,
    checkpoint_transaction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Review final preflight evidence for a future live update without applying it."""

    report = dict(live_application_readiness)
    gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
    target = dict(application_target or {})
    transaction = dict(checkpoint_transaction or {})
    rollback = report.get("rollback_readiness") if isinstance(report.get("rollback_readiness"), Mapping) else {}
    approval = report.get("operator_approval") if isinstance(report.get("operator_approval"), Mapping) else {}
    target_available = bool(target.get("target_available") or target.get("available"))
    target_owned = bool(target.get("owned_by_hecsn", True)) if target else False
    target_mutable = bool(target.get("mutable") or target.get("accepts_bounded_delta"))
    target_sparse = bool(target.get("sparse") or target.get("sparse_transition_weights"))
    target_checkpointed = bool(target.get("checkpointed") or target.get("checkpoint_backed"))
    transaction_opened = bool(transaction.get("pre_update_checkpoint_saved") or transaction.get("opened"))
    transaction_restorable = bool(transaction.get("restore_verified") or transaction.get("restorable"))
    transaction_records_delta = bool(transaction.get("records_shadow_delta") or transaction.get("delta_recorded"))
    required = {
        "live_readiness_available": bool(report.get("available")),
        "live_readiness_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
        "live_readiness_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "readiness_did_not_apply_plasticity": not bool(report.get("applies_plasticity")),
        "readiness_did_not_mutate_runtime": not bool(report.get("mutates_runtime_state")),
        "checkpoint_available": bool(rollback.get("checkpoint_available")),
        "restore_endpoint_available": bool(rollback.get("restore_endpoint_available")),
        "operator_approval_available": bool(approval.get("approved")),
        "application_target_available": target_available,
        "application_target_owned_by_hecsn": target_owned,
        "application_target_mutable": target_mutable,
        "application_target_sparse": target_sparse,
        "application_target_checkpointed": target_checkpointed,
        "checkpoint_transaction_opened": transaction_opened,
        "checkpoint_transaction_restorable": transaction_restorable,
        "checkpoint_transaction_records_delta": transaction_records_delta,
    }
    ready = all(required.values())
    return {
        "artifact_kind": "terminus_snn_language_plasticity_live_application_preflight",
        "surface": "snn_language_plasticity_live_application_preflight.v1",
        "available": bool(report),
        "source": "semantics.spike_language_neurons.plasticity_live_application_preflight",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "live_application_readiness_surface": report.get("surface"),
        "application_target": {
            "available": target_available,
            "surface": target.get("surface"),
            "target_id": target.get("target_id"),
            "owned_by_hecsn": target_owned,
            "mutable": target_mutable,
            "sparse": target_sparse,
            "checkpointed": target_checkpointed,
        },
        "checkpoint_transaction": {
            "pre_update_checkpoint_saved": transaction_opened,
            "checkpoint_path": transaction.get("checkpoint_path") or rollback.get("checkpoint_path"),
            "restore_verified": transaction_restorable,
            "records_shadow_delta": transaction_records_delta,
        },
        "promotion_gate": {
            "status": "ready_for_operator_execution_review" if ready else "blocked_missing_live_application_preflight",
            "eligible_for_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_runtime_training": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_plasticity_application": False,
            "eligible_for_live_application": False,
            "eligible_for_operator_execution_review": ready,
            "requires_operator_approval": True,
            "next_gate": "operator_confirmed_checkpoint_transaction_live_language_plasticity_executor"
            if ready
            else "collect_mutable_target_and_checkpoint_transaction",
            "required_evidence": required,
        },
    }


def build_snn_language_transition_memory_sleep_policy(
    transition_memory_state: Mapping[str, Any],
    *,
    subcortex_sleep_pressure: Mapping[str, Any] | None = None,
    replay_evidence: Mapping[str, Any] | None = None,
    rollout_regeneration_evidence: Mapping[str, Any] | None = None,
    readout_ledger_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend transition-memory homeostasis from active sleep/replay evidence."""

    state = dict(transition_memory_state)
    sleep = dict(subcortex_sleep_pressure or {})
    replay = dict(replay_evidence or {})
    rollout = dict(rollout_regeneration_evidence or {})
    ledger = dict(readout_ledger_evidence or {})
    rollout_gate = rollout.get("promotion_gate") if isinstance(rollout.get("promotion_gate"), Mapping) else {}
    ledger_summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else {}
    transition_count = max(int(_float(state.get("sparse_transition_weight_count"), 0.0)), 0)
    maintenance_count = max(int(_float(state.get("homeostatic_maintenance_count"), 0.0)), 0)
    regeneration_count = max(int(_float(state.get("regeneration_count"), 0.0)), 0)
    regenerated_synapse_count = max(int(_float(state.get("regenerated_synapse_count_total"), 0.0)), 0)
    rollout_event_count = max(int(_float(ledger_summary.get("rollout_event_count"), 0.0)), 0)
    sleep_pressure = min(max(_float(sleep.get("pressure") or sleep.get("sleep_pressure"), 0.0), 0.0), 1.0)
    replay_ready = bool(replay.get("ready") or replay.get("available") or replay.get("replay_ready"))
    rollout_surface = _text(rollout.get("surface"))
    rollout_replay_review_ready = bool(rollout_gate.get("eligible_for_regeneration_permit_request"))
    rollout_permit_ready = bool(
        rollout_surface == "snn_language_readout_rollout_regeneration_permit_request.v1"
        and rollout.get("accepted")
        and rollout_gate.get("eligible_for_regeneration_application")
    )
    rollout_preflight_ready = bool(
        rollout_surface == "snn_language_readout_rollout_regeneration_application_preflight.v1"
        and rollout.get("ready")
        and rollout_gate.get("eligible_for_checkpoint_backed_regeneration_executor")
    )
    rollout_application_applied = bool(
        rollout_surface == "snn_language_readout_rollout_regeneration_application.v1"
        and rollout.get("accepted")
        and rollout.get("mutates_runtime_state")
    )
    pressure_high = sleep_pressure >= 0.5
    memory_present = transition_count > 0
    post_growth_maintenance_due = bool(
        memory_present
        and (rollout_application_applied or regeneration_count > maintenance_count)
        and (pressure_high or replay_ready or regenerated_synapse_count > 0)
    )
    if rollout_preflight_ready:
        action = "review_rollout_regeneration_application"
        suggested_endpoint = "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application"
    elif rollout_permit_ready:
        action = "review_rollout_regeneration_application_preflight"
        suggested_endpoint = "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application-preflight"
    elif rollout_replay_review_ready:
        action = "review_rollout_regeneration_permit_request"
        suggested_endpoint = "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request"
    elif post_growth_maintenance_due or (memory_present and (pressure_high or replay_ready)):
        action = "review_transition_memory_homeostatic_maintenance"
        suggested_endpoint = "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance"
    elif replay_ready or rollout_event_count > 0:
        action = "review_replay_artifact_recording_or_rollout_regeneration"
        suggested_endpoint = "/terminus/snn-language-sequence/transition-memory-replay-artifact/proposal"
    else:
        action = "continue_monitoring_transition_memory"
        suggested_endpoint = None
    recommend = action != "continue_monitoring_transition_memory"
    return {
        "artifact_kind": "terminus_snn_language_transition_memory_sleep_policy",
        "surface": "snn_language_transition_memory_sleep_policy.v1",
        "available": True,
        "source": "semantics.spike_language_neurons.transition_memory_sleep_policy",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "transition_memory": {
            "sparse_transition_weight_count": transition_count,
            "homeostatic_maintenance_count": maintenance_count,
            "regeneration_count": regeneration_count,
            "regenerated_synapse_count_total": regenerated_synapse_count,
        },
        "subcortex_sleep_pressure": {
            "pressure": sleep_pressure,
            "source": sleep.get("source") or "subcortex_sleep_pressure",
            "retired_runtime_dependency": False,
        },
        "replay_evidence": {
            "available": bool(replay),
            "ready": replay_ready,
            "source": replay.get("source"),
        },
        "rollout_regeneration_evidence": {
            "available": bool(rollout),
            "surface": rollout.get("surface"),
            "replay_artifact_review_ready": rollout_replay_review_ready,
            "permit_ready": rollout_permit_ready,
            "application_preflight_ready": rollout_preflight_ready,
            "application_applied": rollout_application_applied,
        },
        "readout_ledger_evidence": {
            "available": bool(ledger),
            "event_count": int(_float(ledger_summary.get("event_count"), 0.0)),
            "rollout_event_count": rollout_event_count,
        },
        "recommendation": {
            "action": action,
            "recommended": recommend,
            "suggested_endpoint": suggested_endpoint,
            "requires_operator_confirmation": True,
            "executable": False,
            "reason_codes": [
                code
                for code, include in (
                    ("transition_memory_present", memory_present),
                    ("subcortex_sleep_pressure_high", pressure_high),
                    ("replay_evidence_ready", replay_ready),
                    ("rollout_replay_artifact_review_ready", rollout_replay_review_ready),
                    ("rollout_regeneration_permit_ready", rollout_permit_ready),
                    ("rollout_regeneration_application_preflight_ready", rollout_preflight_ready),
                    ("rollout_regeneration_application_applied", rollout_application_applied),
                    ("post_growth_homeostatic_maintenance_due", post_growth_maintenance_due),
                    ("readout_rollout_events_available", rollout_event_count > 0),
                )
                if include
            ],
        },
    }


def build_snn_language_transition_memory_regeneration_proposal(
    mismatch_report: Mapping[str, Any],
    transition_memory_state: Mapping[str, Any],
    *,
    replay_evidence: Mapping[str, Any] | None = None,
    locality_radius: int = 2,
    initial_weight: float = 0.02,
    max_new_synapses: int = 8,
) -> dict[str, Any]:
    """Propose bounded local sparse transition regrowth from replay-backed mismatch."""

    mismatch = dict(mismatch_report)
    state = dict(transition_memory_state)
    replay = dict(replay_evidence or {})
    error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
    delta = mismatch.get("sparse_code_delta") if isinstance(mismatch.get("sparse_code_delta"), Mapping) else {}
    radius = max(1, min(int(locality_radius), 8))
    weight = min(max(float(initial_weight), 0.0), 0.25)
    limit = max(1, min(int(max_new_synapses), 32))
    mismatch_score = _float(error.get("mismatch_score"), 0.0)
    replay_window_id = str(replay.get("replay_window_id") or "").strip()
    evidence_hash = str(replay.get("evidence_hash") or "").strip()
    readout_evidence_hashes = [
        str(value)
        for value in list(replay.get("readout_evidence_hashes") or [])
        if str(value)
    ][:64]
    mismatch_hash = hashlib.sha256(
        json.dumps(mismatch, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    replay_ready = (
        bool(replay.get("ready"))
        and bool(replay.get("owned_by_hecsn"))
        and replay.get("artifact_kind") == "terminus_snn_language_transition_memory_regeneration_permit"
        and bool(replay.get("permit_id"))
        and bool(replay_window_id)
        and bool(evidence_hash)
        and replay.get("mismatch_hash") == mismatch_hash
    )
    predicted_only = [
        int(value)
        for value in list(delta.get("predicted_only_indices") or [])
        if isinstance(value, int) and 0 <= int(value) < 64
    ]
    observed_only = [
        int(value)
        for value in list(delta.get("observed_only_indices") or [])
        if isinstance(value, int) and 0 <= int(value) < 64
    ]
    candidates = []
    for pre_index in predicted_only:
        for post_index in observed_only:
            if abs(post_index - pre_index) <= radius:
                candidates.append(
                    {
                        "pre_index": pre_index,
                        "post_index": post_index,
                        "synapse": f"{pre_index}:{post_index}",
                        "initial_weight": weight,
                        "locality_distance": abs(post_index - pre_index),
                    }
                )
    candidates = candidates[:limit]
    ready = bool(mismatch.get("available")) and mismatch_score >= 0.66 and replay_ready and bool(candidates)
    return {
        "artifact_kind": "terminus_snn_language_transition_memory_regeneration_proposal",
        "surface": "snn_language_transition_memory_regeneration_proposal.v1",
        "available": bool(mismatch),
        "source": "semantics.spike_language_neurons.transition_memory_regeneration_proposal",
        "owned_by_hecsn": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "transition_memory_surface": state.get("surface"),
        "mismatch_surface": mismatch.get("surface"),
        "replay_evidence": {
            "available": bool(replay),
            "ready": replay_ready,
            "source": replay.get("source"),
            "artifact_kind": replay.get("artifact_kind"),
            "surface": replay.get("surface"),
            "owned_by_hecsn": bool(replay.get("owned_by_hecsn")),
            "permit_id": replay.get("permit_id"),
            "replay_window_id": replay_window_id or None,
            "evidence_hash": evidence_hash or None,
            "issued_at": replay.get("issued_at"),
            "issued_state_revision": replay.get("issued_state_revision"),
            "operator_id": replay.get("operator_id"),
            "confirmation": replay.get("confirmation"),
            "mismatch_hash": replay.get("mismatch_hash"),
            "pressure_hash": replay.get("pressure_hash"),
            "replay_window_hash": replay.get("replay_window_hash"),
            "replay_window_size": replay.get("replay_window_size"),
            "readout_evidence_hashes": readout_evidence_hashes,
            "replay_artifact_id": replay.get("replay_artifact_id"),
            "replay_artifact_hash": replay.get("replay_artifact_hash"),
            "regeneration_design_hash": replay.get("regeneration_design_hash"),
            "regeneration_design_candidate_count": replay.get(
                "regeneration_design_candidate_count"
            ),
        },
        "regeneration_design": {
            "locality_radius": radius,
            "initial_weight": weight,
            "max_new_synapses": limit,
            "mismatch_score": mismatch_score,
            "candidate_count": len(candidates),
            "candidate_synapses": candidates,
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_regeneration_evidence",
            "eligible_for_regeneration_application": False,
            "eligible_for_language_generation": False,
            "eligible_for_fact_promotion": False,
            "requires_operator_approval": True,
            "next_gate": "operator_confirmed_checkpoint_backed_transition_memory_regeneration"
            if ready
            else "collect_replay_backed_local_mismatch_evidence",
        },
    }


def _language_training_patterns(
    readout_slot_batches: Sequence[Sequence[Mapping[str, Any]]],
    device_report: Mapping[str, Any],
) -> list[tuple[list[int], list[int]]]:
    patterns: list[tuple[list[int], list[int]]] = []
    previous_indices: list[int] | None = None
    for slots in readout_slot_batches:
        readout_slots = [dict(slot) for slot in slots if isinstance(slot, Mapping)]
        decoder_probe = build_spike_language_decoder_probe(
            {
                "readout_slots": readout_slots,
                "device_evidence": device_report,
            }
        )
        sparse_code = (
            decoder_probe.get("sparse_code_evidence")
            if isinstance(decoder_probe.get("sparse_code_evidence"), Mapping)
            else {}
        )
        current_indices = [
            int(value)
            for value in list(sparse_code.get("active_indices") or [])
            if isinstance(value, int)
        ]
        if previous_indices and current_indices:
            patterns.append((previous_indices, current_indices))
        previous_indices = current_indices
    return patterns


def _train_sequence_transition_weights(
    patterns: Sequence[tuple[Sequence[int], Sequence[int]]],
    neuron_count: int,
    device: torch.device,
    *,
    learning_rate: float,
    epochs: int,
) -> torch.Tensor:
    weights = torch.zeros((neuron_count, neuron_count), device=device)
    lr = max(0.0, min(float(learning_rate), 1.0))
    epoch_count = max(1, min(int(epochs), 8))
    for _ in range(epoch_count):
        for current_indices, target_indices in patterns:
            pre = _indices_to_vector(current_indices, neuron_count, device)
            post = _indices_to_vector(target_indices, neuron_count, device)
            weights.mul_(0.995)
            weights.add_(lr * torch.outer(pre, post))
            row_norm = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / row_norm
    return weights


def _indices_to_vector(indices: Sequence[int], neuron_count: int, device: torch.device) -> torch.Tensor:
    vector = torch.zeros(neuron_count, device=device)
    for index in indices:
        vector[int(index) % neuron_count] = 1.0
    total = vector.sum()
    if total > 0:
        vector = vector / total
    return vector


def _persistent_transition_weight_tensor(
    persistent_transition_weights: Mapping[str, Any] | None,
    neuron_count: int,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.zeros((neuron_count, neuron_count), device=device)
    for key, value in dict(persistent_transition_weights or {}).items():
        try:
            raw_pre, raw_post = str(key).split(":", maxsplit=1)
            pre_index = int(raw_pre) % neuron_count
            post_index = int(raw_post) % neuron_count
            weights[pre_index, post_index] += float(value)
        except (TypeError, ValueError):
            continue
    return weights


def _safe_tensor_device(device: str) -> torch.device:
    normalized = str(device or "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError):
        return torch.device("cpu")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mismatch_band(value: float) -> str:
    if value <= 0.25:
        return "low"
    if value <= 0.65:
        return "medium"
    return "high"


__all__ = [
    "SpikeLanguageNeuronAdapter",
    "build_spike_language_plasticity_application_design",
    "build_spike_language_neuron_adapter",
    "evaluate_spike_language_adapter_heldout",
    "build_spike_language_plasticity_shadow_delta",
    "build_snn_language_transition_memory_sleep_policy",
    "build_snn_language_transition_memory_regeneration_proposal",
    "evaluate_spike_language_plasticity_live_application_readiness",
    "evaluate_spike_language_plasticity_live_application_preflight",
    "evaluate_spike_language_sequence_mismatch",
    "evaluate_spike_language_plasticity_shadow_application",
    "evaluate_spike_language_trainer_dry_run",
    "evaluate_spike_language_plasticity_replay",
    "evaluate_snn_language_readout_rollout_replay",
    "build_spike_language_plasticity_pressure",
    "run_spike_language_plasticity_replay_experiment",
    "run_spike_language_plasticity_trial",
    "generate_snn_language_readout_draft",
    "rollout_snn_language_readout_candidate",
    "build_snn_language_transition_memory_prediction_evaluation",
    "predict_spike_language_sequence",
    "run_spike_language_trainer_dry_run",
]
