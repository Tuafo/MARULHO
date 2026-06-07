from __future__ import annotations

from marulho.semantics.spike_language_decoder import (
    build_spike_language_decoder_probe,
)
from marulho.semantics.spike_language_neurons import (
    build_spike_language_neuron_adapter,
    evaluate_spike_language_sequence_mismatch,
    predict_spike_language_sequence,
)


def _slot(label: str) -> dict[str, object]:
    return {
        "label": label,
        "grounded": True,
        "pressure_band": "low",
    }


def test_sequence_prediction_preserves_dynamic_index_above_63() -> None:
    report = predict_spike_language_sequence(
        [[_slot("test")], [_slot("next")]],
        [_slot("test")],
        {"device": "cpu", "source": "unit"},
        persistent_transition_weights={"4:64": 0.9},
        language_neuron_count=66,
        top_k=8,
    )

    assert report["language_neuron_count"] == 66
    assert 64 in report["prediction"]["predicted_sparse_indices"]
    assert report["device_evidence"]["tensor_device"] == "cpu"


def test_dynamic_mismatch_and_neuron_adapter_use_live_capacity() -> None:
    prediction = {
        "surface": "snn_language_sequence_prediction_probe.v1",
        "available": True,
        "language_neuron_count": 66,
        "prediction": {"predicted_sparse_indices": [64]},
        "device_evidence": {"tensor_device": "cpu"},
    }
    observed = [_slot("x" * 64)]
    mismatch = evaluate_spike_language_sequence_mismatch(
        prediction,
        observed,
    )
    decoder = build_spike_language_decoder_probe(
        {
            "readout_slots": observed,
            "device_evidence": {"device": "cpu"},
            "code_dim": 66,
        }
    )
    adapter = build_spike_language_neuron_adapter(decoder)

    assert mismatch["language_neuron_count"] == 66
    assert 64 in mismatch["sparse_code_delta"]["matched_indices"]
    assert adapter["neuron_dynamics"]["neuron_count"] == 66
