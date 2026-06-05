"""Grounding evidence diagnostics for language-facing readouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingDiagnostics:
    """Term-level support evidence for a grounded language-facing result."""

    kind: str
    target: str
    target_terms: tuple[str, ...] = ()
    matched_target_terms: tuple[str, ...] = ()
    evidence_supported_terms: tuple[str, ...] = ()
    grounded_evidence_count: int = 0
    response_coverage: float = 0.0
    evidence_coverage: float = 0.0
    evidence_alignment: float = 0.0
    alignment_score: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "target_terms": list(self.target_terms),
            "matched_target_terms": list(self.matched_target_terms),
            "evidence_supported_terms": list(self.evidence_supported_terms),
            "grounded_evidence_count": int(self.grounded_evidence_count),
            "response_coverage": float(self.response_coverage),
            "evidence_coverage": float(self.evidence_coverage),
            "evidence_alignment": float(self.evidence_alignment),
            "alignment_score": float(self.alignment_score),
            "fallback_used": bool(self.fallback_used),
        }
