"""Tests for the canonical Subcortex Cognitive Signal state."""

from __future__ import annotations

from hecsn.semantics.cognitive_signal import CognitiveSignalState


def test_cognitive_signal_state_clamps_and_serializes_control_payload() -> None:
    signal = CognitiveSignalState.from_mapping(
        {
            "schema_version": "",
            "source": "subcortex.runtime",
            "sampled_at": -10.0,
            "prediction_error_mean": 1.5,
            "prediction_error_max": -0.5,
            "predictive_confidence_mean": 0.25,
            "predictive_confidence_min": 2.0,
            "dopamine": 1.2,
            "norepinephrine": -0.2,
            "recent_concepts": ["reef chemistry", "thermal stress", "unused-a", "unused-b", "unused-c", "unused-d", "unused-e"],
            "concept_candidates": [
                {
                    "label": "reef chemistry",
                    "top_terms": ["reef", "carbonate", "unused", "extra", "drop"],
                    "match_count": 3,
                    "uncertainty": 1.4,
                    "temporal_coherence": -0.1,
                    "example_windows": ["Reef carbonate balance shifted.", "Thermal stress increased.", "drop"],
                }
            ],
        }
    )

    payload = signal.to_mapping()

    assert payload["schema_version"] == "cognitive_signal.v1"
    assert payload["source"] == "subcortex.runtime"
    assert payload["sampled_at"] == 0.0
    assert payload["prediction_error_mean"] == 1.0
    assert payload["prediction_error_max"] == 0.0
    assert payload["predictive_confidence_mean"] == 0.25
    assert payload["predictive_confidence_min"] == 1.0
    assert payload["dopamine"] == 1.0
    assert payload["norepinephrine"] == 0.0
    assert payload["recent_concepts"] == [
        "reef chemistry",
        "thermal stress",
        "unused-a",
        "unused-b",
        "unused-c",
        "unused-d",
    ]
    assert payload["concept_candidates"][0]["top_terms"] == ["reef", "carbonate", "unused", "extra"]
    assert payload["concept_candidates"][0]["uncertainty"] == 1.0
    assert payload["concept_candidates"][0]["temporal_coherence"] == 0.0
    assert payload["concept_candidates"][0]["example_windows"] == [
        "Reef carbonate balance shifted.",
        "Thermal stress increased.",
    ]
