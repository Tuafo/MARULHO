"""Tests for canonical grounding diagnostics."""

from __future__ import annotations

from hecsn.semantics.grounding_diagnostics import GroundingDiagnostics


def test_grounding_diagnostics_serializes_tuple_fields_as_lists() -> None:
    diagnostics = GroundingDiagnostics(
        kind="query",
        target="How do reefs handle thermal stress?",
        target_terms=("reefs", "thermal", "stress"),
        matched_target_terms=("reefs", "stress"),
        evidence_supported_terms=("reefs",),
        grounded_evidence_count=2,
        response_coverage=0.66,
        evidence_coverage=0.33,
        evidence_alignment=0.5,
        alignment_score=0.59,
        fallback_used=True,
    )

    assert diagnostics.to_dict() == {
        "kind": "query",
        "target": "How do reefs handle thermal stress?",
        "target_terms": ["reefs", "thermal", "stress"],
        "matched_target_terms": ["reefs", "stress"],
        "evidence_supported_terms": ["reefs"],
        "grounded_evidence_count": 2,
        "response_coverage": 0.66,
        "evidence_coverage": 0.33,
        "evidence_alignment": 0.5,
        "alignment_score": 0.59,
        "fallback_used": True,
    }
