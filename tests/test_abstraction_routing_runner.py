from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.abstraction_routing_runner import run_abstraction_routing_benchmark


class AbstractionRoutingRunnerTests(unittest.TestCase):
    def test_abstraction_routing_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_abstraction_routing_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["abstraction_routing_gate"]["pass"])
        self.assertTrue(summary["runtime_scope"]["supports_first_class_abstraction"])
        self.assertGreater(summary["routing_bias"]["gain_margin"], 0.0)
        self.assertTrue(summary["routing_bias"]["equal_score_probe"]["winner_changed"])
        self.assertEqual(summary["routing_bias"]["equal_score_probe"]["biased_winner"], 1)


if __name__ == "__main__":
    unittest.main()
