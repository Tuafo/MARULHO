from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.terminus_long_horizon_runner import run_terminus_long_horizon_benchmark


class TerminusLongHorizonBenchmarkTests(unittest.TestCase):
    def test_long_horizon_terminus_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_terminus_long_horizon_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["long_horizon_gate"]["pass"])
        self.assertGreaterEqual(summary["metrics"]["supported_topic_coverage"], 1.0)
        self.assertGreaterEqual(summary["metrics"]["revisit_provider_hit_rate"], 1.0)
        self.assertGreaterEqual(summary["metrics"]["concept_stability_mean"], 0.20)
        self.assertGreaterEqual(summary["metrics"]["revisit_retention_rate"], 1.0)

        ranked = summary["final_runtime"]["autonomy"]["provider_curriculum"]["ranked_providers"]
        providers = {str(item["provider"]): item for item in ranked}
        self.assertIn("submarine", providers["wikipedia"]["topic_families"])
        self.assertIn("octopus", providers["openalex"]["topic_families"])

        submarine_revisit = next(
            step
            for step in summary["steps"]
            if bool(step.get("revisit")) and str(step.get("topic")) == "submarine"
        )
        self.assertTrue(submarine_revisit["provider_hit"])
        self.assertTrue(submarine_revisit["supported"])
        self.assertNotIn("trim", submarine_revisit["unsupported_terms"])
        self.assertIn("trim", str(submarine_revisit["response_text"]).lower())


if __name__ == "__main__":
    unittest.main()
