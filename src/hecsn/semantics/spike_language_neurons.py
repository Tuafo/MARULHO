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
    "build_spike_language_neuron_adapter",
    "evaluate_spike_language_adapter_heldout",
    "evaluate_spike_language_sequence_mismatch",
    "evaluate_spike_language_trainer_dry_run",
    "predict_spike_language_sequence",
    "run_spike_language_trainer_dry_run",
]
