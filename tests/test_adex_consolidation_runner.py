from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.adex_consolidation_runner import run_adex_consolidation_benchmark


class AdExConsolidationRunnerTests(unittest.TestCase):
    def test_adex_consolidation_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_adex_consolidation_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["adex_consolidation_gate"]["pass"])
        self.assertTrue(summary["backends"]["proxy"]["memory_consolidation_gate"]["pass"])
        self.assertTrue(summary["backends"]["adex"]["memory_consolidation_gate"]["pass"])
        self.assertTrue(summary["backends"]["adex"]["finite_model_state"])
        self.assertGreater(summary["backends"]["adex"]["mean_post_spike_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
