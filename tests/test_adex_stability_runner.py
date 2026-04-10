from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.adex_stability_runner import run_adex_stability_benchmark


class AdExStabilityRunnerTests(unittest.TestCase):
    def test_adex_stability_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_adex_stability_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["adex_stability_gate"]["pass"])
        self.assertEqual(summary["metrics"]["case_count"], 5)
        self.assertEqual(summary["metrics"]["finite_case_fraction"], 1.0)
        self.assertEqual(summary["metrics"]["spiking_case_fraction"], 1.0)
        self.assertTrue(any(case["name"] == "stress_exc" and case["pass"] for case in summary["cases"]))


if __name__ == "__main__":
    unittest.main()
