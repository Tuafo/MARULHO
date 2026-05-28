"""HECSN-owned spike-language decoder probe.

The probe turns existing Subcortex Spike Readout Evidence into a sparse tensor
code and support report. It is intentionally non-generative: labels come from
grounded readout slots, not from sampling or an external language model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


class SpikeLanguageDecoderProbe:
    """Small deterministic probe for future SNN-native language decoding."""

    def __init__(self, *, code_dim: int = 32, max_slots: int = 4) -> None:
        self.code_dim = max(8, int(code_dim))
        self.max_slots = max(1, int(max_slots))

    def evaluate(self, spike_readout_evidence: Mapping[str, Any]) -> dict[str, Any]:
        device_report = (
            spike_readout_evidence.get("device_evidence")
            if isinstance(spike_readout_evidence.get("device_evidence"), Mapping)
            else {}
        )
        target_device = _safe_tensor_device(str(device_report.get("device") or "cpu"))
        readout_slots = [
            slot
            for slot in list(spike_readout_evidence.get("readout_slots") or [])[: self.max_slots]
            if isinstance(slot, Mapping)
        ]
        code = torch.zeros(self.code_dim, device=target_device)
        labels: list[str] = []
        grounded_count = 0
        pressure_indices: list[int] = []
        for index, slot in enumerate(readout_slots):
            label = _text(slot.get("label"))
            if label:
                labels.append(label)
            if bool(slot.get("grounded")):
                grounded_count += 1
            base_index = (index * 7 + len(label)) % self.code_dim
            pressure_index = (base_index + _pressure_offset(_text(slot.get("pressure_band")))) % self.code_dim
            code[base_index] = 1.0
            code[pressure_index] = 1.0
            pressure_indices.append(int(pressure_index))

        active_count = int(torch.count_nonzero(code).item())
        sparsity = 1.0 - (active_count / max(1, int(code.numel())))
        grounded_fraction = grounded_count / max(1, len(readout_slots))
        supported = bool(readout_slots) and grounded_fraction >= 0.5 and active_count > 0
        return {
            "artifact_kind": "terminus_hecsn_spike_language_decoder_probe",
            "surface": "snn_language_decoder_probe_evidence.v1",
            "available": bool(readout_slots),
            "source": "semantics.spike_language_decoder_probe",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "executable": False,
            "decodes_text": False,
            "trains": False,
            "mutates_runtime_state": False,
            "device_evidence": {
                "requested_device": str(device_report.get("device") or "unknown"),
                "tensor_device": str(code.device),
                "cuda_tensor": bool(code.is_cuda),
                "device_source": device_report.get("source"),
            },
            "sparsity_evidence": {
                "code_dim": int(code.numel()),
                "active_count": active_count,
                "mean_sparsity": float(sparsity),
                "target_min_sparsity": 0.85,
                "meets_sparse_readout_floor": bool(sparsity >= 0.85),
            },
            "support_evidence": {
                "readout_slot_count": len(readout_slots),
                "grounded_slot_count": grounded_count,
                "unsupported_slot_count": max(0, len(readout_slots) - grounded_count),
                "grounded_fraction": float(grounded_fraction),
                "grounded_slot_labels": labels[: self.max_slots],
                "supported": supported,
            },
            "symbolic_trace": {
                "labels": labels[: self.max_slots],
                "pressure_indices": pressure_indices,
            },
            "promotion_constraints": {
                "eligible_for_language_generation": False,
                "requires_training_loop": True,
                "requires_grounding_support": True,
                "requires_evaluation_dataset": True,
                "requires_operator_approval": True,
            },
        }


def build_spike_language_decoder_probe(spike_readout_evidence: Mapping[str, Any]) -> dict[str, Any]:
    return SpikeLanguageDecoderProbe().evaluate(spike_readout_evidence)


def _pressure_offset(pressure_band: str) -> int:
    if pressure_band == "high":
        return 3
    if pressure_band == "medium":
        return 2
    return 1


def _safe_tensor_device(device: str) -> torch.device:
    normalized = str(device or "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except (RuntimeError, TypeError):
        return torch.device("cpu")


def _text(value: Any) -> str:
    return str(value or "").strip()


__all__ = ["SpikeLanguageDecoderProbe", "build_spike_language_decoder_probe"]
