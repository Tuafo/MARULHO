from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.adex_backend_runner import run_adex_backend_benchmark


class AdExBackendRunnerTests(unittest.TestCase):
    def test_adex_backend_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_adex_backend_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["adex_backend_gate"]["pass"])
        self.assertTrue(summary["backends"]["proxy"]["finite_parameters"])
        self.assertTrue(summary["backends"]["adex"]["finite_parameters"])
        self.assertTrue(summary["backends"]["adex"]["uses_adex_post_spikes"])
        self.assertGreater(summary["backends"]["adex"]["mean_post_spike_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
