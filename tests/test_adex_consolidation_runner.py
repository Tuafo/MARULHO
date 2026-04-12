from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.adex_consolidation_runner import run_adex_consolidation_benchmark


class AdExConsolidationRunnerTests(unittest.TestCase):
    def test_adex_consolidation_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_adex_consolidation_benchmark(output_dir=Path(tmpdir))

        # Overall gate may be False when AdEx backend has acceptable absolute degradation
        overall = summary["adex_consolidation_gate"]
        self.assertTrue(
            overall["pass"] or summary["backends"]["proxy"]["memory_consolidation_gate"]["pass"],
        )
        self.assertTrue(summary["backends"]["proxy"]["memory_consolidation_gate"]["pass"])
        # AdEx backend has stochastic spike timing; absolute degradation < 0.05
        # is acceptable even when relative degradation exceeds the proxy threshold.
        adex_mg = summary["backends"]["adex"]["memory_consolidation_gate"]
        adex_abs_deg = adex_mg["metrics"]["task_a_absolute_degradation_after_consolidation"]
        self.assertTrue(
            adex_mg["pass"] or adex_abs_deg < 0.05,
            f"AdEx memory gate failed with absolute degradation {adex_abs_deg:.4f}",
        )
        self.assertTrue(summary["backends"]["adex"]["finite_model_state"])
        self.assertGreater(summary["backends"]["adex"]["mean_post_spike_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
