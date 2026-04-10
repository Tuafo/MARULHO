from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.geometric_curiosity_runner import run_geometric_curiosity_benchmark


class GeometricCuriosityRunnerTests(unittest.TestCase):
    def test_geometric_curiosity_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_geometric_curiosity_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["geometric_curiosity_gate"]["pass"])
        self.assertTrue(summary["focus_plan"]["retrieval_queries"])
        self.assertTrue(summary["focus_plan"]["geometric_gaps"])


if __name__ == "__main__":
    unittest.main()
