"""Tests for golden output registry (Stage 0C)."""

from __future__ import annotations

import unittest

from hecsn.evaluation.golden_outputs import (
    GoldenOutputRegistry,
    GoldenRecord,
    STAGE_0_GOLDEN,
)


class GoldenRecordTests(unittest.TestCase):
    def test_check_within_tolerance(self) -> None:
        r = GoldenRecord(name="metric", value=0.5, tolerance=0.05)
        self.assertTrue(r.check(0.50))
        self.assertTrue(r.check(0.54))
        self.assertTrue(r.check(0.46))

    def test_check_outside_tolerance(self) -> None:
        r = GoldenRecord(name="metric", value=0.5, tolerance=0.05)
        self.assertFalse(r.check(0.60))
        self.assertFalse(r.check(0.40))

    def test_check_exact(self) -> None:
        r = GoldenRecord(name="metric", value=1.0, tolerance=0.0)
        self.assertTrue(r.check(1.0))
        self.assertFalse(r.check(1.001))


class GoldenOutputRegistryTests(unittest.TestCase):
    def test_register_and_check(self) -> None:
        reg = GoldenOutputRegistry()
        reg.register("acc", 0.7, tolerance=0.05)
        result = reg.check_all({"acc": 0.72})
        self.assertTrue(result["acc"]["pass"])
        self.assertAlmostEqual(result["acc"]["delta"], 0.02)

    def test_check_missing_key_ignored(self) -> None:
        reg = GoldenOutputRegistry()
        reg.register("acc", 0.7)
        result = reg.check_all({"other": 0.5})
        self.assertEqual(len(result), 0)

    def test_summary_all_pass(self) -> None:
        reg = GoldenOutputRegistry()
        reg.register("a", 1.0, tolerance=0.1)
        reg.register("b", 2.0, tolerance=0.1)
        s = reg.summary({"a": 1.05, "b": 2.05})
        self.assertTrue(s["all_pass"])
        self.assertEqual(s["passed"], 2)

    def test_summary_partial_fail(self) -> None:
        reg = GoldenOutputRegistry()
        reg.register("a", 1.0, tolerance=0.01)
        reg.register("b", 2.0, tolerance=0.01)
        s = reg.summary({"a": 1.0, "b": 3.0})
        self.assertFalse(s["all_pass"])
        self.assertEqual(s["failed"], 1)

    def test_stage_0_golden_has_records(self) -> None:
        """Verify the pre-populated Stage 0 registry exists."""
        self.assertGreater(len(STAGE_0_GOLDEN.records), 0)
        self.assertIn("silhouette", STAGE_0_GOLDEN.records)
        self.assertIn("grounding_probe_50_accuracy", STAGE_0_GOLDEN.records)

    def test_stage_0_golden_values_reasonable(self) -> None:
        """Verify pre-populated values are within plausible ranges."""
        for name, record in STAGE_0_GOLDEN.records.items():
            self.assertIsInstance(record.value, float)
            self.assertGreater(record.tolerance, 0.0)


if __name__ == "__main__":
    unittest.main()
