from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.training.emergence_evaluation_runner import run_emergence_evaluation_benchmark


class EmergenceEvaluationRunnerTests(unittest.TestCase):
    def test_emergence_evaluation_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_emergence_evaluation_benchmark(output_dir=Path(tmpdir))

        self.assertTrue(summary["emergence_evaluation_gate"]["pass"])
        self.assertTrue(summary["feedback_emergence_gate"]["pass"])
        self.assertGreaterEqual(summary["metrics"]["temporal_coherence_mean"], 0.95)
        self.assertGreaterEqual(summary["metrics"]["grounded_query_accuracy"], 0.85)
        self.assertGreaterEqual(summary["metrics"]["compositional_query_accuracy"], 0.60)
        self.assertGreaterEqual(summary["metrics"]["phase_a_interference_retention"], 0.95)
        self.assertGreaterEqual(summary["metrics"]["phase_a_final_retention"], 0.95)
        self.assertTrue(summary["meaning_grounding"]["mixed_world"]["gate_pass"])
        self.assertTrue(summary["novelty_coverage"]["healthy_coverage"])
        self.assertEqual(len(summary["compositionality"]["cases"]), 3)
        self.assertTrue(summary["feedback_label_free_levels"]["structural_coherence"]["temporal_pass"])
        self.assertTrue(summary["feedback_label_free_levels"]["compositionality"]["direct_metric_available"])
        self.assertTrue(summary["feedback_label_free_levels"]["grounding_probe"]["direct_metric_available"])
        self.assertTrue(summary["feedback_label_free_levels"]["novelty_coverage"]["direct_metric_available"])
        self.assertTrue(summary["feedback_label_free_levels"]["grounding_probe"]["pass"])
        self.assertTrue(summary["feedback_label_free_levels"]["novelty_coverage"]["pass"])
        self.assertGreater(summary["compositionality"]["direct_sample_count"], 0)
        self.assertGreater(summary["grounding_probe"]["direct_sample_count"], 0)
        self.assertEqual(summary["grounding_probe"]["concrete_count"], 25)
        self.assertEqual(summary["grounding_probe"]["abstract_count"], 25)
        self.assertIn("concreteness_gap", summary["grounding_probe"])
        self.assertGreater(summary["direct_novelty_coverage"]["sample_count"], 0)
        self.assertEqual(summary["direct_novelty_coverage"]["stream_unit"], "learned_chunk")
        self.assertTrue(summary["baseline_comparison"]["representation"]["representation_collapse_detected"])
        self.assertFalse(summary["baseline_comparison"]["mechanism"]["distributional_clustering_pass"])
        self.assertTrue(summary["supporting_scaffolds"]["routing_scale"]["gate_pass"])


if __name__ == "__main__":
    unittest.main()
