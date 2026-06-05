"""Subcortex cognitive-signal state primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CognitiveSignalState:
    """Recent Subcortex control signals for pressure, focus, and confidence."""

    schema_version: str = "cognitive_signal.v1"
    source: str = "subcortex"
    sampled_at: float = 0.0
    prediction_error_mean: float = 0.0
    prediction_error_max: float = 0.0
    predictive_confidence_mean: float = 0.5
    predictive_confidence_min: float = 0.5
    dopamine: float = 0.0
    norepinephrine: float = 0.0
    recent_concepts: tuple[str, ...] = ()
    concept_candidates: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, payload: Any | None) -> "CognitiveSignalState":
        if isinstance(payload, cls):
            return payload
        payload = payload or {}
        if not hasattr(payload, "get"):
            payload = {}
        concepts = payload.get("recent_concepts", ())
        if not isinstance(concepts, (list, tuple)):
            concepts = ()
        raw_candidates = payload.get("concept_candidates", ())
        concept_candidates: list[dict[str, Any]] = []
        if isinstance(raw_candidates, (list, tuple)):
            for item in raw_candidates[:8]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                top_terms = [
                    str(term).strip()
                    for term in list(item.get("top_terms") or [])[:4]
                    if str(term).strip()
                ]
                example_windows = [
                    str(text).strip()
                    for text in list(item.get("example_windows") or [])[:2]
                    if str(text).strip()
                ]
                concept_candidates.append(
                    {
                        "label": label,
                        "top_terms": top_terms,
                        "match_count": int(item.get("match_count", item.get("observations", 0)) or 0),
                        "observations": int(item.get("observations", item.get("match_count", 0)) or 0),
                        "uncertainty": max(0.0, min(1.0, float(item.get("uncertainty", 1.0)))),
                        "temporal_coherence": max(
                            0.0,
                            min(1.0, float(item.get("temporal_coherence", 0.0))),
                        ),
                        "example_windows": example_windows,
                    }
                )
        return cls(
            schema_version=str(payload.get("schema_version", "cognitive_signal.v1") or "cognitive_signal.v1"),
            source=str(payload.get("source", "subcortex") or "subcortex"),
            sampled_at=max(0.0, float(payload.get("sampled_at", 0.0) or 0.0)),
            prediction_error_mean=max(0.0, min(1.0, float(payload.get("prediction_error_mean", 0.0)))),
            prediction_error_max=max(0.0, min(1.0, float(payload.get("prediction_error_max", 0.0)))),
            predictive_confidence_mean=max(
                0.0,
                min(1.0, float(payload.get("predictive_confidence_mean", 0.5))),
            ),
            predictive_confidence_min=max(
                0.0,
                min(1.0, float(payload.get("predictive_confidence_min", 0.5))),
            ),
            dopamine=max(0.0, min(1.0, float(payload.get("dopamine", 0.0)))),
            norepinephrine=max(0.0, min(1.0, float(payload.get("norepinephrine", 0.0)))),
            recent_concepts=tuple(str(concept) for concept in concepts[:6] if concept),
            concept_candidates=tuple(concept_candidates),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "sampled_at": float(self.sampled_at),
            "prediction_error_mean": float(self.prediction_error_mean),
            "prediction_error_max": float(self.prediction_error_max),
            "predictive_confidence_mean": float(self.predictive_confidence_mean),
            "predictive_confidence_min": float(self.predictive_confidence_min),
            "dopamine": float(self.dopamine),
            "norepinephrine": float(self.norepinephrine),
            "recent_concepts": list(self.recent_concepts),
            "concept_candidates": [dict(item) for item in self.concept_candidates],
        }
