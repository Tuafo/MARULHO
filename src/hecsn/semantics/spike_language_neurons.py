"""HECSN-owned spike-language neuron adapter evidence.

The adapter consumes the local decoder probe's sparse code and produces
bounded neuron-dynamics evidence. It is intentionally not a generator.
"""

from __future__ import annotations

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
    current_vector = _indices_to_vector(current_indices, neuron_count, device)
    logits = current_vector @ weights
    requested_k = max(1, min(int(top_k), 16, neuron_count))
    predicted = torch.topk(logits, k=requested_k)
    predicted_indices = [int(value) for value in predicted.indices.detach().cpu().tolist()]
    predicted_strengths = [float(value) for value in predicted.values.detach().cpu().tolist()]
    nonzero_weight_count = int(torch.count_nonzero(weights).item())
    total_weight_count = int(weights.numel())
    weight_sparsity = 1.0 - (nonzero_weight_count / max(1, total_weight_count))
    support_strength = float(sum(max(0.0, value) for value in predicted_strengths))
    available = bool(train_patterns and current_indices)
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
        "current_sparse_code": {
            "active_index_count": len(current_indices),
            "active_indices": current_indices[:16],
            "mean_sparsity": _float(current_sparse.get("mean_sparsity"), 1.0),
        },
        "prediction": {
            "predicted_sparse_indices": predicted_indices,
            "predicted_sparse_strengths": predicted_strengths,
            "support_strength": support_strength,
            "top_k": requested_k,
        },
        "training_evidence": {
            "rule": "local_hebbian_outer_product_with_row_normalization",
            "training_transition_count": len(train_patterns),
            "learning_rate": max(0.0, min(float(learning_rate), 1.0)),
            "epochs": max(1, min(int(epochs), 8)),
            "weight_sparsity": float(weight_sparsity),
            "target_min_weight_sparsity": 0.85,
        },
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


def _safe_tensor_device(device: str) -> torch.device:
    normalized = str(device or "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError):
        return torch.device("cpu")


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
    "evaluate_spike_language_sequence_mismatch",
    "evaluate_spike_language_plasticity_shadow_application",
    "evaluate_spike_language_trainer_dry_run",
    "evaluate_spike_language_plasticity_replay",
    "build_spike_language_plasticity_pressure",
    "run_spike_language_plasticity_replay_experiment",
    "run_spike_language_plasticity_trial",
    "predict_spike_language_sequence",
    "run_spike_language_trainer_dry_run",
]
