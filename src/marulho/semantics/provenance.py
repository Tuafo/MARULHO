"""Shared provenance vocabulary for grounded Subcortex evidence."""

from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    """How evidence was acquired; determines default trust weighting."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    DREAMED = "dreamed"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"

    @property
    def trust_weight(self) -> float:
        return {
            Provenance.OBSERVED: 0.8,
            Provenance.INFERRED: 0.6,
            Provenance.DREAMED: 0.3,
            Provenance.VERIFIED: 1.0,
            Provenance.CONTRADICTED: 0.1,
        }[self]
