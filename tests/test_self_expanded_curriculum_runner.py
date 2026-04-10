from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.self_expanded_curriculum_runner import run_self_expanded_curriculum_benchmark


class SelfExpandedCurriculumRunnerTests(unittest.TestCase):
    def test_self_expanded_curriculum_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_self_expanded_curriculum_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["self_expanded_curriculum_gate"]["pass"])
        self.assertTrue(summary["focus_plan"]["geometric_gaps"])
        self.assertGreaterEqual(
            int(summary["second_spec"]["catalog_queries_per_provider"]),
            int(summary["first_spec"]["catalog_queries_per_provider"]),
        )
        self.assertGreaterEqual(
            int(summary["second_spec"]["catalog_query_family_budget_bonus"]),
            1,
        )
        self.assertTrue(summary["second_spec"]["catalog_provider_query_families"]["wikipedia"])


if __name__ == "__main__":
    unittest.main()
