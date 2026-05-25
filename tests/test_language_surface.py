from __future__ import annotations

import unittest

from hecsn.semantics import (
    build_subcortical_deliberation_surface,
    build_subcortical_self_repair_surface,
)


class SubcorticalDeliberationSurfaceTests(unittest.TestCase):
    def test_high_prediction_error_yields_stabilize_candidate(self) -> None:
        surface = build_subcortical_deliberation_surface(
            {
                "prediction_error_mean": 0.42,
                "prediction_error_max": 0.71,
                "predictive_confidence_mean": 0.51,
                "predictive_confidence_min": 0.34,
                "dopamine": 0.1,
                "norepinephrine": 0.2,
                "recent_concepts": ["coral thermal memory"],
                "concept_candidates": [
                    {
                        "label": "coral thermal memory",
                        "top_terms": ["coral", "thermal", "memory"],
                        "observations": 4,
                        "uncertainty": 0.3,
                        "temporal_coherence": 0.6,
                    }
                ],
            }
        )

        self.assertEqual(surface["surface"], "subcortical_control_candidates.v1")
        self.assertEqual(surface["control_hint"], "stabilize_prediction")
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertFalse(surface["retired_runtime_dependency"])
        self.assertEqual(surface["candidates"][0]["intent"], "stabilize_prediction")
        self.assertIn("coral thermal memory", surface["candidates"][0]["candidate_text"])
        self.assertNotIn("prompt", surface["candidates"][0])
        self.assertIn("prediction_error_mean", surface["candidates"][0]["grounding"])
        for replay_key in ("candidate_id", "target_type", "suggested_endpoint", "suggested_input", "reason_codes"):
            self.assertNotIn(replay_key, surface["candidates"][0])
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "ready_for_replay_review")
        self.assertTrue(surface["candidates"][0]["promotion_gate"]["eligible_for_replay_review"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_fact_promotion"])
        self.assertEqual(surface["promotion_summary"]["ready_for_replay_review"], 3)
        self.assertFalse(surface["promotion_summary"]["eligible_for_action"])

    def test_missing_focus_yields_collect_grounding(self) -> None:
        surface = build_subcortical_deliberation_surface(
            {
                "prediction_error_mean": 0.04,
                "prediction_error_max": 0.08,
                "predictive_confidence_mean": 0.60,
                "predictive_confidence_min": 0.50,
                "recent_concepts": [],
                "concept_candidates": [],
            }
        )

        self.assertEqual(surface["control_hint"], "collect_grounding")
        self.assertEqual(surface["candidates"][0]["intent"], "collect_grounding")
        self.assertEqual(surface["grounding"]["concept_focus"], None)
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "blocked_missing_grounding")
        self.assertEqual(surface["promotion_summary"]["blocked_missing_grounding"], 1)
        self.assertLessEqual(len(surface["candidates"]), 3)


class SubcorticalSelfRepairSurfaceTests(unittest.TestCase):
    def test_silent_spike_health_yields_repair_review_candidate(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "silent_risk",
                "silent_fraction": 0.91,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.4,
                "correlation_evidence_available": False,
                "correlation": {"status": "insufficient_window", "sample_count": 2},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertFalse(surface["retired_runtime_dependency"])
        self.assertEqual(surface["candidates"][0]["intent"], "review_column_revival")
        self.assertIn("promotion_gate", surface["candidates"][0])
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "ready_for_replay_review")
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(surface["promotion_summary"]["eligible_for_structural_mutation"])

    def test_overcorrelated_spike_health_yields_decorrelation_candidate(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "sparse_responsive",
                "silent_fraction": 0.0,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.0,
                "correlation_evidence_available": True,
                "correlation": {
                    "status": "overcorrelated_risk",
                    "sample_count": 8,
                    "active_columns": 3,
                    "valid_pairs": 6,
                    "mean_abs_offdiag_correlation": 0.91,
                },
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["candidates"][0]["intent"], "review_decorrelation_or_prune")
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["next_gate"], "deep_sleep_or_replay_repair_gate")
        self.assertEqual(surface["promotion_summary"]["ready_for_replay_review"], 1)

    def test_insufficient_spike_window_stays_monitor_only(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "sparse_responsive",
                "silent_fraction": 0.0,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.0,
                "correlation_evidence_available": False,
                "correlation": {"status": "insufficient_window", "sample_count": 0},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["candidates"][0]["intent"], "monitor_spike_health")
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "insufficient_evidence")
        self.assertEqual(surface["promotion_summary"]["insufficient_evidence"], 1)


if __name__ == "__main__":
    unittest.main()
