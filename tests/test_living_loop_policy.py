"""Dedicated tests for the Policy Scoring module (living_loop_policy.py).

Covers:
- PolicyScore round-trip and defaults
- WorldModelLiteSummary.from_payload round-trip
- WorldModelLiteSummary.from_records with known prediction/action/consolidation counts
- PolicyActuatorRecommendation round-trip
- build_policy_actuator_status for known living-loop payloads
- Re-export verification: symbols still importable from living_loop
- Policy action selection logic (contradictions > pending > memory > latency > uncertainty > healthy)
"""
from __future__ import annotations

import unittest
from typing import Any, Mapping

from hecsn.cortex.episodic_memory import Provenance
from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    ActionVerificationRecord,
    ConsolidationRecord,
    ConsolidationStatus,
    PredictionRecord,
    PredictionStatus,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Import from the new Policy Scoring module
# ---------------------------------------------------------------------------
from hecsn.service.living_loop_policy import (
    POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
    POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
    POLICY_ACTUATOR_SCHEMA_VERSION,
    PolicyActuatorRecommendation,
    PolicyScore,
    WorldModelLiteSummary,
    build_policy_actuator_status,
    _coerce_feedback_telemetry,
    _policy_count,
    _policy_float,
    _policy_mapping,
)


class TestPolicyScoreRoundTrip(unittest.TestCase):
    """PolicyScore.from_payload / to_payload round-trip and defaults."""

    def test_default_policy_score(self) -> None:
        ps = PolicyScore()
        self.assertEqual(ps.information_gain, 0.0)
        self.assertEqual(ps.goal_progress, 0.0)
        self.assertEqual(ps.cost, 0.0)
        self.assertEqual(ps.risk, 0.0)
        self.assertEqual(ps.budget_use, 0.0)
        self.assertEqual(ps.uncertainty, 1.0)
        self.assertEqual(ps.recommended_next_action, "observe_or_execute_grounded_action")
        self.assertEqual(ps.components, {})

    def test_from_payload_none(self) -> None:
        ps = PolicyScore.from_payload(None)
        self.assertEqual(ps, PolicyScore())

    def test_from_payload_clamps_values(self) -> None:
        ps = PolicyScore.from_payload({"information_gain": 1.5, "risk": -0.2, "uncertainty": 0.3})
        self.assertEqual(ps.information_gain, 1.0)
        self.assertEqual(ps.risk, 0.0)
        self.assertEqual(ps.uncertainty, 0.3)

    def test_from_payload_cleans_recommended_next_action(self) -> None:
        ps = PolicyScore.from_payload({"recommended_next_action": "  investigate  "})
        self.assertEqual(ps.recommended_next_action, "investigate")

    def test_from_payload_empty_string_gives_default_action(self) -> None:
        ps = PolicyScore.from_payload({"recommended_next_action": ""})
        self.assertEqual(ps.recommended_next_action, "observe_or_execute_grounded_action")

    def test_to_payload_round_trip(self) -> None:
        original = PolicyScore(
            information_gain=0.5,
            goal_progress=0.6,
            cost=0.3,
            risk=0.2,
            budget_use=0.4,
            uncertainty=0.1,
            recommended_next_action="continue_current_policy",
            components={"alpha": 1},
        )
        payload = original.to_payload()
        restored = PolicyScore.from_payload(payload)
        self.assertAlmostEqual(restored.information_gain, 0.5)
        self.assertAlmostEqual(restored.goal_progress, 0.6)
        self.assertAlmostEqual(restored.cost, 0.3)
        self.assertAlmostEqual(restored.risk, 0.2)
        self.assertAlmostEqual(restored.budget_use, 0.4)
        self.assertAlmostEqual(restored.uncertainty, 0.1)
        self.assertEqual(restored.recommended_next_action, "continue_current_policy")
        self.assertEqual(restored.components, {"alpha": 1})


class TestWorldModelLiteSummaryFromRecords(unittest.TestCase):
    """WorldModelLiteSummary.from_records with known prediction/action/consolidation counts."""

    def test_empty_records_give_defaults(self) -> None:
        wml = WorldModelLiteSummary.from_records()
        self.assertEqual(wml.prediction_count, 0)
        self.assertEqual(wml.fulfilled_count, 0)
        self.assertEqual(wml.contradicted_count, 0)
        self.assertEqual(wml.verification_count, 0)
        self.assertEqual(wml.prediction_accuracy, 0.0)
        # With no records, uncertainty should be high (1.0)
        self.assertEqual(wml.uncertainty, 1.0)

    def test_known_prediction_counts(self) -> None:
        predictions = (
            PredictionRecord("p1", "A", status=PredictionStatus.FULFILLED, provenance=Provenance.VERIFIED, confidence=0.9),
            PredictionRecord("p2", "B", status=PredictionStatus.CONTRADICTED, provenance=Provenance.CONTRADICTED, confidence=0.7),
            PredictionRecord("p3", "C", status=PredictionStatus.PENDING, provenance=Provenance.INFERRED, confidence=0.5),
            PredictionRecord("p4", "D", status=PredictionStatus.UNKNOWN, provenance=Provenance.INFERRED, confidence=0.0),
        )
        wml = WorldModelLiteSummary.from_records(predictions=predictions)
        self.assertEqual(wml.prediction_count, 4)
        self.assertEqual(wml.fulfilled_count, 1)
        self.assertEqual(wml.contradicted_count, 1)
        self.assertEqual(wml.pending_count, 1)
        self.assertEqual(wml.unknown_count, 1)
        self.assertEqual(wml.evaluated_prediction_count, 2)  # fulfilled + contradicted

    def test_known_action_verification_counts(self) -> None:
        actions = (
            ActionExecutionRecord(
                action_id="a1",
                action_type="search",
                execution_status=ActionExecutionStatus.EXECUTED,
                prediction=PredictionRecord("p1", "Find evidence"),
                verification=ActionVerificationRecord(
                    "v1", VerificationStatus.VERIFIED, success=True, confidence=0.8, contradiction=False
                ),
            ),
            ActionExecutionRecord(
                action_id="a2",
                action_type="read",
                execution_status=ActionExecutionStatus.FAILED,
                prediction=PredictionRecord("p2", "Read file"),
                verification=ActionVerificationRecord(
                    "v2", VerificationStatus.CONTRADICTED, success=False, confidence=0.6, contradiction=True
                ),
            ),
            ActionExecutionRecord(
                action_id="a3",
                action_type="fetch",
                execution_status=ActionExecutionStatus.EXECUTED,
                prediction=PredictionRecord("p3", "Fetch data"),
                verification=ActionVerificationRecord(
                    "v3", VerificationStatus.UNVERIFIED, success=False, confidence=0.4, contradiction=False
                ),
            ),
        )
        wml = WorldModelLiteSummary.from_records(actions=actions)
        # Verification count excludes UNKNOWN-status only; VERIFIED, CONTRADICTED, UNVERIFIED all count
        self.assertEqual(wml.verification_count, 3)
        self.assertEqual(wml.verified_action_count, 1)
        self.assertEqual(wml.contradicted_action_count, 1)
        self.assertEqual(wml.unverified_action_count, 1)  # UNVERIFIED status

    def test_known_consolidation_counts(self) -> None:
        consolidations = (
            ConsolidationRecord(
                "c1", ConsolidationStatus.CREDITED, "response", "test query",
                credit_events=3, penalty_events=1, forgiveness_events=2, aggregate_count=5,
                trajectory_net_score=0.8,
            ),
            ConsolidationRecord(
                "c2", ConsolidationStatus.PENALIZED, "response", "test query 2",
                credit_events=0, penalty_events=2, forgiveness_events=0, aggregate_count=3,
                trajectory_net_score=-0.3,
            ),
        )
        wml = WorldModelLiteSummary.from_records(consolidations=consolidations)
        self.assertEqual(wml.components["credit_events"], 3)
        self.assertEqual(wml.components["penalty_events"], 3)
        self.assertEqual(wml.components["forgiveness_events"], 2)

    def test_from_records_recommended_action_contradictions(self) -> None:
        predictions = (
            PredictionRecord("p1", "X", status=PredictionStatus.CONTRADICTED, provenance=Provenance.CONTRADICTED),
        )
        wml = WorldModelLiteSummary.from_records(predictions=predictions)
        self.assertEqual(wml.recommended_next_action, "investigate_contradictions")

    def test_from_records_recommended_action_pending(self) -> None:
        predictions = (
            PredictionRecord("p1", "X", status=PredictionStatus.PENDING, provenance=Provenance.INFERRED),
        )
        wml = WorldModelLiteSummary.from_records(predictions=predictions)
        self.assertEqual(wml.recommended_next_action, "verify_pending_predictions")

    def test_from_records_recommended_action_empty(self) -> None:
        wml = WorldModelLiteSummary.from_records()
        self.assertEqual(wml.recommended_next_action, "observe_or_execute_grounded_action")

    def test_from_payload_round_trip(self) -> None:
        original = WorldModelLiteSummary(
            prediction_count=10,
            fulfilled_count=7,
            contradicted_count=2,
            pending_count=1,
            unknown_count=0,
            evaluated_prediction_count=9,
            prediction_accuracy=0.78,
            verification_count=5,
            verified_action_count=4,
            contradicted_action_count=1,
            unverified_action_count=0,
            verification_success_rate=0.8,
            contradiction_rate=0.2,
            information_gain=0.5,
            goal_progress=0.6,
            cost=0.3,
            risk=0.2,
            budget_use=0.4,
            uncertainty=0.1,
            recommended_next_action="continue_current_policy",
            policy_score=PolicyScore(
                information_gain=0.5,
                goal_progress=0.6,
                cost=0.3,
                risk=0.2,
                budget_use=0.4,
                uncertainty=0.1,
                recommended_next_action="continue_current_policy",
            ),
        )
        payload = original.to_payload()
        restored = WorldModelLiteSummary.from_payload(payload)
        self.assertEqual(restored.prediction_count, 10)
        self.assertEqual(restored.fulfilled_count, 7)
        self.assertEqual(restored.contradicted_count, 2)
        self.assertAlmostEqual(restored.prediction_accuracy, 0.78)
        self.assertEqual(restored.recommended_next_action, "continue_current_policy")
        self.assertAlmostEqual(restored.policy_score.information_gain, 0.5)


class TestPolicyActuatorRecommendationRoundTrip(unittest.TestCase):
    """PolicyActuatorRecommendation.to_payload round-trip."""

    def test_to_payload_includes_schema_version(self) -> None:
        rec = PolicyActuatorRecommendation(
            recommendation="Test",
            action="continue_current_policy",
            reasons=({"code": "test", "detail": "test"},),
            risk=0.1,
            expected_information_gain=0.5,
            expected_goal_progress=0.4,
            expected_cost=0.2,
            uncertainty=0.3,
        )
        payload = rec.to_payload()
        self.assertEqual(payload["schema_version"], POLICY_ACTUATOR_SCHEMA_VERSION)
        self.assertTrue(payload["advisory"])
        self.assertFalse(payload["executable"])

    def test_to_payload_created_at_auto_generated(self) -> None:
        rec = PolicyActuatorRecommendation(
            recommendation="Test",
            action="continue_current_policy",
            reasons=(),
            risk=0.0,
            expected_information_gain=0.0,
            expected_goal_progress=0.0,
            expected_cost=0.0,
            uncertainty=0.0,
        )
        payload = rec.to_payload()
        self.assertTrue(payload["created_at"])  # auto-generated ISO timestamp


class TestBuildPolicyActuatorStatus(unittest.TestCase):
    """build_policy_actuator_status output for known living-loop payloads."""

    def test_contradictions_highest_priority(self) -> None:
        """Contradictions take priority over memory pressure and latency."""
        result = build_policy_actuator_status(
            {
                "world_model_lite": {
                    "contradicted_count": 1,
                    "contradicted_action_count": 1,
                    "risk": 0.2,
                    "uncertainty": 0.1,
                    "policy_score": {"cost": 0.1, "budget_use": 0.1},
                },
                "feedback_summary": {
                    "feedback_count": 1,
                    "contradicted_count": 1,
                    "recent_feedback": [
                        {
                            "target_type": "action",
                            "target_id": "act-1",
                            "verdict": "contradicted",
                            "applied_status": "contradicted",
                        }
                    ],
                },
                "actions": [
                    {
                        "action_id": "act-1",
                        "action_type": "workspace_read",
                        "verification": {"status": "contradicted", "contradiction": True},
                    }
                ],
                "memory_health": {"fill_ratio": 0.95},
                "benchmark_telemetry": {
                    "endpoint_latency_ms": {"query": {"avg_ms": 2500.0, "max_ms": 3000.0}},
                },
            },
            cortex_snapshot={"drives": {"fatigue": 0.95}},
        )
        self.assertEqual(result.action, "investigate_contradictions")
        self.assertEqual(result.target_action_id, "act-1")
        self.assertTrue(result.advisory)
        self.assertFalse(result.executable)

    def test_pending_evidence_before_sleep(self) -> None:
        """Pending evidence takes priority over memory/sleep pressure."""
        result = build_policy_actuator_status(
            {
                "world_model_lite": {
                    "pending_count": 1,
                    "unverified_action_count": 1,
                    "uncertainty": 0.2,
                    "policy_score": {"cost": 0.1, "budget_use": 0.1},
                },
                "feedback_summary": {"unverified_count": 1},
                "actions": [
                    {
                        "action_id": "act-pending",
                        "action_type": "workspace_search",
                        "verification": {"status": "unverified"},
                    }
                ],
                "memory_health": {"fill_ratio": 0.99},
            },
            cortex_snapshot={"drives": {"fatigue": 0.9}},
        )
        self.assertEqual(result.action, "verify_pending_evidence")
        self.assertEqual(result.target_action_id, "act-pending")

    def test_latency_pressure_triggers_reduce_scope(self) -> None:
        """High latency triggers reduce_scope_or_wait before collecting evidence."""
        result = build_policy_actuator_status(
            {
                "world_model_lite": {
                    "unknown_count": 1,
                    "uncertainty": 0.95,
                    "policy_score": {"cost": 0.85, "budget_use": 0.2},
                },
                "feedback_summary": {},
                "benchmark_telemetry": {
                    "endpoint_latency_ms": {"respond": {"avg_ms": 1200.0, "max_ms": 1800.0}},
                },
                "uncertain_domains": [{"domain": "cats", "total_uncertain_signals": 1}],
            }
        )
        self.assertEqual(result.action, "reduce_scope_or_wait")

    def test_empty_payload_gives_continue_current_policy(self) -> None:
        result = build_policy_actuator_status({})
        self.assertEqual(result.action, "continue_current_policy")
        self.assertTrue(result.advisory)

    def test_memory_pressure_triggers_consolidate_or_sleep(self) -> None:
        """High memory fill triggers consolidate_or_sleep."""
        result = build_policy_actuator_status(
            {
                "world_model_lite": {
                    "uncertainty": 0.1,
                    "policy_score": {"cost": 0.1, "budget_use": 0.1},
                },
                "memory_health": {"fill_ratio": 0.95},
            }
        )
        self.assertEqual(result.action, "consolidate_or_sleep")

    def test_uncertainty_triggers_collect_more_evidence(self) -> None:
        """High uncertainty (without contradictions, pending, memory, or latency pressure) triggers collect_more_evidence."""
        result = build_policy_actuator_status(
            {
                "world_model_lite": {
                    "uncertainty": 0.70,
                    "policy_score": {"cost": 0.1, "budget_use": 0.1},
                },
                "memory_health": {"fill_ratio": 0.3},
            }
        )
        self.assertEqual(result.action, "collect_more_evidence")


class TestPolicyConstants(unittest.TestCase):
    """Policy module constants."""

    def test_schema_version_is_int(self) -> None:
        self.assertIsInstance(POLICY_ACTUATOR_SCHEMA_VERSION, int)

    def test_latency_thresholds_positive(self) -> None:
        self.assertGreater(POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS, 0)
        self.assertGreater(POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS, 0)


class TestPolicySharedHelpers(unittest.TestCase):
    """Shared policy helpers: _policy_count, _policy_mapping, _policy_float, _coerce_feedback_telemetry."""

    def test_policy_count_extracts_non_negative_int(self) -> None:
        self.assertEqual(_policy_count({"a": 5}, "a"), 5)
        self.assertEqual(_policy_count({"a": 0}, "a"), 0)
        self.assertEqual(_policy_count({"a": -3}, "a"), 0)
        self.assertEqual(_policy_count({"a": None}, "a"), 0)
        self.assertEqual(_policy_count({}, "a"), 0)

    def test_policy_mapping_returns_dict_for_mapping(self) -> None:
        result = _policy_mapping({"x": 1})
        self.assertEqual(result, {"x": 1})

    def test_policy_mapping_returns_empty_dict_for_non_mapping(self) -> None:
        result = _policy_mapping("not a mapping")
        self.assertEqual(result, {})

    def test_policy_float_returns_first_valid(self) -> None:
        self.assertAlmostEqual(_policy_float(None, "bad", 0.7), 0.7)

    def test_policy_float_returns_zero_on_all_none(self) -> None:
        self.assertAlmostEqual(_policy_float(None, None), 0.0)

    def test_coerce_feedback_telemetry_none(self) -> None:
        result = _coerce_feedback_telemetry(None)
        self.assertEqual(result["feedback_count"], 0)
        self.assertEqual(result["grounding_impact"], "none")


class TestReExportFromLivingLoop(unittest.TestCase):
    """Verify backward compatibility: symbols still importable from living_loop."""

    def test_policy_score_importable(self) -> None:
        from hecsn.service.living_loop import PolicyScore as LL_PS
        self.assertIs(LL_PS, PolicyScore)

    def test_world_model_lite_summary_importable(self) -> None:
        from hecsn.service.living_loop import WorldModelLiteSummary as LL_WML
        self.assertIs(LL_WML, WorldModelLiteSummary)

    def test_policy_actuator_recommendation_importable(self) -> None:
        from hecsn.service.living_loop import PolicyActuatorRecommendation as LL_PAR
        self.assertIs(LL_PAR, PolicyActuatorRecommendation)

    def test_build_policy_actuator_status_importable(self) -> None:
        from hecsn.service.living_loop import build_policy_actuator_status as LL_BPAS
        self.assertIs(LL_BPAS, build_policy_actuator_status)


if __name__ == "__main__":
    unittest.main()
