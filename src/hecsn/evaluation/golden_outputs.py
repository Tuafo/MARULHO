"""Golden output registry for regression detection (Stage 0C).

Records known-good metric values from deterministic seed runs.
Each golden record has a tolerance — if a future run deviates beyond
tolerance, something changed (numpy version, algorithm, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoldenRecord:
    """A single golden metric value with tolerance."""

    name: str
    value: float
    tolerance: float = 0.05
    seed: int = 7
    description: str = ""

    def check(self, actual: float) -> bool:
        """Return True if actual is within tolerance of golden value."""
        return abs(actual - self.value) <= self.tolerance


@dataclass
class GoldenOutputRegistry:
    """Registry of golden outputs for deterministic regression detection."""

    records: dict[str, GoldenRecord] = field(default_factory=dict)

    def register(
        self,
        name: str,
        value: float,
        tolerance: float = 0.05,
        seed: int = 7,
        description: str = "",
    ) -> None:
        self.records[name] = GoldenRecord(
            name=name,
            value=value,
            tolerance=tolerance,
            seed=seed,
            description=description,
        )

    def check_all(self, actuals: dict[str, float]) -> dict[str, dict[str, Any]]:
        """Check all registered goldens against actual values.

        Returns dict mapping name -> {golden, actual, pass, delta}.
        """
        results: dict[str, dict[str, Any]] = {}
        for name, record in self.records.items():
            if name in actuals:
                actual = actuals[name]
                results[name] = {
                    "golden": record.value,
                    "actual": actual,
                    "pass": record.check(actual),
                    "delta": actual - record.value,
                    "tolerance": record.tolerance,
                }
        return results

    def summary(self, actuals: dict[str, float]) -> dict[str, Any]:
        checks = self.check_all(actuals)
        n_pass = sum(1 for v in checks.values() if v["pass"])
        n_total = len(checks)
        return {
            "total_checked": n_total,
            "passed": n_pass,
            "failed": n_total - n_pass,
            "all_pass": n_pass == n_total,
            "details": checks,
        }


# Pre-populated registry with known-good values from seed=7, numpy 2.4.4
# These are updated after each verified full-suite green run.
STAGE_0_GOLDEN = GoldenOutputRegistry()

# Emergence evaluation (seed=7)
STAGE_0_GOLDEN.register(
    "silhouette",
    0.675,
    tolerance=0.10,
    description="Clustering silhouette score from maintained probe.",
)
STAGE_0_GOLDEN.register(
    "dbi",
    0.304,
    tolerance=0.15,
    description="Davies-Bouldin index from maintained probe.",
)
STAGE_0_GOLDEN.register(
    "temporal_coherence_mean",
    0.9916,
    tolerance=0.05,
    description="Temporal coherence from maintained probe.",
)
STAGE_0_GOLDEN.register(
    "routing_key_between_score",
    0.997,
    tolerance=0.05,
    description="Routing key between-score from maintained probe.",
)
STAGE_0_GOLDEN.register(
    "semantic_triple_accuracy",
    0.714286,
    tolerance=0.10,
    description="Semantic triple accuracy (7-triple) from maintained probe.",
)
STAGE_0_GOLDEN.register(
    "terminal_novelty_rate",
    0.0994,
    tolerance=0.10,
    description="Terminal novelty rate from maintained probe.",
)

# Grounding probe 50-triple (seed=7)
STAGE_0_GOLDEN.register(
    "grounding_probe_50_accuracy",
    0.62,
    tolerance=0.10,
    description="50-triple grounding probe total accuracy (text-only baseline).",
)
STAGE_0_GOLDEN.register(
    "grounding_probe_50_concreteness_gap",
    -0.04,
    tolerance=0.15,
    description="50-triple concreteness gap (text-only, expected near zero or negative).",
)
