"""Dedicated tests for the Replay Planning module (living_loop_replay.py).

Covers:
- ReplayCandidate round-trip and construction
- ReplayPlan round-trip and construction
- build_replay_plan with known payloads and candidate ranking order
- replay_candidate_safety_flags computation
- Priority score calculation
- _replay_priority_score with known component weights
- Re-export verification: symbols still importable from living_loop
- REPLAY_PLAN_PRIORITY_WEIGHTS and REPLAY_REASON_PRECEDENCE constants
- _coerce_replay_sample_summary round-trip
- _default_replay_sample_safety_flags structure
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
import unittest
from typing import Any, Mapping

from marulho.service.runtime_evidence import RuntimeEvidenceReporter
from marulho.service.living_loop_replay import (
    REPLAY_PLAN_DEFAULT_LIMIT,
    REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
    REPLAY_PLAN_FEEDBACK_WINDOW_LIMIT,
    REPLAY_PLAN_MAX_LIMIT,
    REPLAY_PLAN_PRIORITY_RULES_VERSION,
    REPLAY_PLAN_PRIORITY_WEIGHTS,
    REPLAY_PLAN_SCHEMA_VERSION,
    REPLAY_PLAN_SOURCE_WINDOW_LIMIT,
    REPLAY_PLAN_SOURCE_WINDOW_POLICY,
    REPLAY_REASON_PRECEDENCE,
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    ReplayCandidate,
    ReplayPlan,
    build_replay_plan,
    replay_candidate_safety_flags,
    _coerce_replay_sample_summary,
    _default_replay_sample_safety_flags,
    _replay_action_for_reasons,
    _replay_endpoint_for_action,
    _replay_priority_score,
    _replay_reason_precedence,
    _replay_unique_reasons,
)


class _ReplayWindowProbeSequence(SequenceABC[Mapping[str, Any]]):
    def __init__(self, items: list[Mapping[str, Any]]) -> None:
        self._items = list(items)
        self.slice_calls = 0
        self.integer_getitem_calls = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        if isinstance(index, slice):
            self.slice_calls += 1
            return self._items[index]
        self.integer_getitem_calls += 1
        return self._items[index]


class TestReplayCandidateRoundTrip(unittest.TestCase):
    """ReplayCandidate construction and to_payload round-trip."""

    def test_basic_round_trip(self) -> None:
        candidate = ReplayCandidate(
            candidate_id="replay-test-123",
            rank=1,
            target_type="runtime_episode",
            target_id="episode-1",
            target_ids=("episode-1",),
            operation="respond",
            created_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            reason_codes=("contradicted_feedback",),
            priority_score=150.0,
            priority_components={"safety": 1.0, "feedback": 0.7},
            suggested_consolidation_action="review_contradiction",
            suggested_endpoint="/terminus/runtime-feedback",
            suggested_input={"target_type": "runtime_episode"},
            summary="Test candidate.",
            provenance={"provenance": "observed"},
            risk=0.5,
            uncertainty=0.6,
            latency={"latency_ms": 1800.0},
            memory_health={"status": "available"},
            feedback={"contradicted_count": 1},
            policy={"action": "investigate_contradictions"},
        )
        payload = candidate.to_payload()
        self.assertEqual(payload["candidate_id"], "replay-test-123")
        self.assertEqual(payload["rank"], 1)
        self.assertEqual(payload["target_type"], "runtime_episode")
        self.assertEqual(payload["target_id"], "episode-1")
        self.assertEqual(payload["target_ids"], ["episode-1"])
        self.assertEqual(payload["reason_codes"], ["contradicted_feedback"])
        self.assertAlmostEqual(payload["priority_score"], 150.0)
        self.assertTrue(payload["priority_components"]["safety"], 1.0)
        self.assertEqual(payload["suggested_consolidation_action"], "review_contradiction")

    def test_target_ids_serialized_as_list(self) -> None:
        candidate = ReplayCandidate(
            candidate_id="rc-1", rank=2, target_type="action", target_id="a1",
            target_ids=("a1", "a2"), operation="search",
            created_at="2026-01-01T00:00:00+00:00", completed_at="",
            reason_codes=("unverified_action",), priority_score=50.0,
            priority_components={}, suggested_consolidation_action="verify_pending_evidence",
            suggested_endpoint="/terminus/runtime-feedback", suggested_input={},
            summary="Test", provenance={}, risk=0.0, uncertainty=0.3,
            latency={}, memory_health={}, feedback={}, policy={},
        )
        payload = candidate.to_payload()
        self.assertIsInstance(payload["target_ids"], list)
        self.assertEqual(payload["target_ids"], ["a1", "a2"])


class TestReplayPlanRoundTrip(unittest.TestCase):
    """ReplayPlan construction and to_payload round-trip."""

    def test_basic_round_trip(self) -> None:
        plan = ReplayPlan(
            generated_at="2026-01-01T00:00:00+00:00",
            limit=20,
            state_revision=7,
            token_count=42,
            snapshot_counts={"runtime_episodes": 1, "actions": 0, "predictions": 1, "feedback": 0, "uncertain_domains": 0},
            priority_weights=dict(REPLAY_PLAN_PRIORITY_WEIGHTS),
            plan_reason_codes=("contradicted_feedback",),
            candidates=(),
        )
        payload = plan.to_payload()
        self.assertEqual(payload["schema_version"], REPLAY_PLAN_SCHEMA_VERSION)
        self.assertTrue(payload["advisory"])
        self.assertFalse(payload["executable"])
        self.assertEqual(payload["endpoint"], "/terminus/replay-plan")
        self.assertEqual(payload["priority_rules_version"], REPLAY_PLAN_PRIORITY_RULES_VERSION)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["limit"], 20)


class TestReplayPlanSelfRepairIsolation(unittest.TestCase):
    """Self-repair gate artifacts must not become replay execution candidates."""

    def test_self_repair_gate_artifact_is_ignored_by_replay_plan(self) -> None:
        living_loop = {
            "state_revision": 7,
            "token_count": 42,
            "subcortical_self_repair_candidates": {
                "schema_version": 1,
                "artifact_kind": "terminus_subcortical_self_repair_gate_plan",
                "surface": "subcortical_self_repair_candidates.v1",
                "advisory": True,
                "executable": False,
                "promotion_gate": {
                    "status": "ready_for_replay_review",
                    "next_gate": "deep_sleep_or_replay_repair_gate",
                    "eligible_for_structural_mutation": False,
                },
                "candidates": [
                    {
                        "intent": "review_decorrelation_or_prune",
                        "candidate_text": "Review decorrelation before pruning.",
                        "promotion_gate": {
                            "next_gate": "deep_sleep_or_replay_repair_gate",
                            "eligible_for_structural_mutation": False,
                        },
                    }
                ],
            },
            "runtime_episodes": [
                {
                    "episode_id": "ep-1",
                    "operation": "respond",
                    "status": "ok",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": "2026-01-01T00:00:01+00:00",
                    "verification": {"status": "verified", "confidence": 0.9},
                    "prediction": {"status": "verified", "confidence": 0.9},
                }
            ],
        }

        plan = build_replay_plan(living_loop, limit=10).to_payload()

        for candidate in plan["candidates"]:
            self.assertNotIn(
                candidate["target_type"],
                {"self_repair", "subcortical_self_repair", "repair_review"},
            )
            self.assertNotIn("candidate_text", candidate)
            self.assertNotEqual(
                candidate.get("operation"),
                "deep_sleep_or_replay_repair_gate",
            )
            self.assertNotIn("review_decorrelation_or_prune", candidate.get("reason_codes", []))

    def test_plan_with_candidates(self) -> None:
        candidate = ReplayCandidate(
            candidate_id="rc-1", rank=1, target_type="runtime_episode", target_id="e1",
            target_ids=("e1",), operation="respond",
            created_at="2026-01-01T00:00:00+00:00", completed_at="",
            reason_codes=("contradicted_feedback",), priority_score=150.0,
            priority_components={}, suggested_consolidation_action="review_contradiction",
            suggested_endpoint="/terminus/runtime-feedback", suggested_input={},
            summary="Test", provenance={}, risk=0.5, uncertainty=0.6,
            latency={}, memory_health={}, feedback={}, policy={},
        )
        plan = ReplayPlan(
            generated_at="2026-01-01T00:00:00+00:00",
            limit=5,
            state_revision=0,
            token_count=0,
            snapshot_counts={},
            priority_weights={},
            plan_reason_codes=("contradicted_feedback",),
            candidates=(candidate,),
        )
        payload = plan.to_payload()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["candidates"][0]["candidate_id"], "rc-1")


class TestReplayPriorityScore(unittest.TestCase):
    """_replay_priority_score calculation."""

    def test_all_components_zero(self) -> None:
        score = _replay_priority_score({
            "safety": 0.0, "feedback": 0.0, "uncertainty": 0.0,
            "memory_pressure": 0.0, "latency_pressure": 0.0,
            "policy_pressure": 0.0, "provenance_gap": 0.0, "recency_rank": 0.0,
        })
        self.assertAlmostEqual(score, 0.0)

    def test_safety_component_dominates(self) -> None:
        score = _replay_priority_score({
            "safety": 1.0, "feedback": 0.0, "uncertainty": 0.0,
            "memory_pressure": 0.0, "latency_pressure": 0.0,
            "policy_pressure": 0.0, "provenance_gap": 0.0, "recency_rank": 0.0,
        })
        self.assertAlmostEqual(score, 100.0)

    def test_full_priority_components(self) -> None:
        components = {
            "safety": 1.0, "feedback": 1.0, "uncertainty": 1.0,
            "memory_pressure": 1.0, "latency_pressure": 1.0,
            "policy_pressure": 1.0, "provenance_gap": 1.0, "recency_rank": 1.0,
        }
        expected = sum(
            REPLAY_PLAN_PRIORITY_WEIGHTS[key] * 1.0 for key in REPLAY_PLAN_PRIORITY_WEIGHTS
        )
        score = _replay_priority_score(components)
        self.assertAlmostEqual(score, round(expected, 6))


class TestReplayCandidateSafetyFlags(unittest.TestCase):
    """replay_candidate_safety_flags computation."""

    def test_contradicted_feedback_is_non_factual(self) -> None:
        flags = replay_candidate_safety_flags({
            "reason_codes": ["contradicted_feedback"],
            "suggested_consolidation_action": "review_contradiction",
            "feedback": {"contradicted_count": 1},
            "provenance": {"provenance": "observed"},
        })
        self.assertTrue(flags["audit_only"])
        self.assertTrue(flags["not_promoted"])
        self.assertFalse(flags["promoted_to_verified_fact"])
        self.assertTrue(flags["non_factual"])
        self.assertTrue(flags["negative_lesson"])
        self.assertFalse(flags["dreamed_or_synthetic"])

    def test_dreamed_or_synthetic(self) -> None:
        flags = replay_candidate_safety_flags({
            "reason_codes": [],
            "suggested_consolidation_action": "",
            "feedback": {},
            "provenance": {"provenance": "synthetic_dream"},
        })
        self.assertTrue(flags["dreamed_or_synthetic"])
        self.assertTrue(flags["non_factual"])

    def test_healthy_candidate_not_non_factual(self) -> None:
        flags = replay_candidate_safety_flags({
            "reason_codes": ["healthy_grounded_state"],
            "suggested_consolidation_action": "continue_observing",
            "feedback": {"verified_count": 1},
            "provenance": {"provenance": "observed"},
            "status": "verified",
        })
        self.assertTrue(flags["audit_only"])
        self.assertFalse(flags["non_factual"])
        self.assertFalse(flags["negative_lesson"])
        self.assertFalse(flags["dreamed_or_synthetic"])

    def test_safety_boundaries_included(self) -> None:
        flags = replay_candidate_safety_flags({"reason_codes": [], "feedback": {}})
        self.assertEqual(flags["safety_boundaries"], list(REPLAY_SAMPLE_SAFETY_BOUNDARIES))
        self.assertIn("no_training", flags["safety_boundaries"])

    def test_failed_runtime_is_negative_lesson(self) -> None:
        flags = replay_candidate_safety_flags({
            "reason_codes": ["failed_runtime_episode"],
            "suggested_consolidation_action": "",
            "feedback": {},
            "verification_status": "failed",
        })
        self.assertTrue(flags["negative_lesson"])
        self.assertTrue(flags["non_factual"])


class TestBuildReplayPlan(unittest.TestCase):
    """build_replay_plan with known payloads and candidate ranking order."""

    def test_empty_payload_produces_healthy_grounded_state(self) -> None:
        plan = build_replay_plan({}, limit=5, created_at="2026-01-01T00:00:00+00:00")
        payload = plan.to_payload()
        self.assertEqual(payload["schema_version"], 1)
        self.assertTrue(payload["advisory"])
        self.assertFalse(payload["executable"])
        self.assertEqual(payload["endpoint"], "/terminus/replay-plan")
        self.assertGreaterEqual(payload["count"], 1)
        # The default healthy candidate
        self.assertEqual(payload["candidates"][0]["target_type"], "policy_decision")
        self.assertIn("healthy_grounded_state", payload["candidates"][0]["reason_codes"])

    def test_contradicted_episode_ranks_highest(self) -> None:
        plan = build_replay_plan(
            {
                "state_revision": 7,
                "token_count": 42,
                "runtime_episode_count": 1,
                "prediction_count": 1,
                "action_count": 0,
                "runtime_episodes": [
                    {
                        "episode_id": "episode-contradicted",
                        "operation": "respond",
                        "status": "succeeded",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:00:01+00:00",
                        "latency_ms": 1800.0,
                        "prediction": {"proposed_answer": "Cats always fly.", "confidence": 0.2},
                        "verification": {"status": "contradicted", "confidence": 0.8, "contradiction": True},
                        "feedback": [
                            {
                                "feedback_id": "fb-1",
                                "created_at": "2026-01-01T00:00:02+00:00",
                                "target_type": "runtime_episode",
                                "target_id": "episode-contradicted",
                                "verdict": "contradicted",
                                "applied_status": "contradicted",
                                "confidence": 0.95,
                                "summary": "Manual review found the answer was wrong.",
                                "corrected_output": {"response_text": "Cats do not fly."},
                            },
                        ],
                    },
                ],
                "feedback_summary": {
                    "feedback_count": 1,
                    "contradicted_count": 1,
                    "recent_feedback": [
                        {
                            "feedback_id": "fb-1",
                            "created_at": "2026-01-01T00:00:02+00:00",
                            "target_type": "runtime_episode",
                            "target_id": "episode-contradicted",
                            "verdict": "contradicted",
                            "applied_status": "contradicted",
                            "confidence": 0.95,
                            "summary": "Manual review found the answer was wrong.",
                            "has_corrected_output": True,
                        },
                    ],
                },
                "world_model_lite": {"uncertainty": 0.4, "policy_score": {"cost": 0.1}},
                "memory_health": {"status": "available", "fill_ratio": 0.25},
                "benchmark_telemetry": {"endpoint_latency_ms": {"respond": {"avg_ms": 1800.0, "max_ms": 1800.0}}},
                "policy_decision": {
                    "action": "investigate_contradictions",
                    "recommendation": "Investigate contradictions.",
                    "risk": 0.75,
                    "uncertainty": 0.4,
                    "reasons": [{"code": "contradicted_feedback", "detail": "test"}],
                    "advisory": True,
                    "executable": False,
                },
            },
            limit=5,
            created_at="2026-01-01T00:00:03+00:00",
        ).to_payload()

        self.assertEqual(plan["snapshot_counts"]["runtime_episodes"], 1)
        self.assertGreaterEqual(plan["count"], 1)
        top = plan["candidates"][0]
        self.assertEqual(top["target_type"], "runtime_episode")
        self.assertEqual(top["target_id"], "episode-contradicted")
        self.assertEqual(top["suggested_consolidation_action"], "review_contradiction")
        self.assertIn("contradicted_feedback", top["reason_codes"])
        self.assertIn("corrected_output_available", top["reason_codes"])
        self.assertIn("high_latency", top["reason_codes"])
        self.assertGreater(top["priority_score"], 100.0)
        self.assertEqual(top["feedback"]["contradicted_count"], 1)
        self.assertTrue(top["suggested_input"]["operator_review_required"])

    def test_candidate_ranking_by_priority_score(self) -> None:
        """Contradicted candidates should rank higher than unverified ones."""
        plan = build_replay_plan(
            {
                "runtime_episodes": [
                    {
                        "episode_id": "ep-unverified",
                        "operation": "query",
                        "status": "succeeded",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "verification": {"status": "unverified", "confidence": 0.5},
                        "prediction": {"confidence": 0.5},
                    },
                ],
                "actions": [
                    {
                        "action_id": "act-contradicted",
                        "action_type": "search",
                        "recorded_at": "2026-01-01T00:00:00+00:00",
                        "verification": {"status": "contradicted", "contradiction": True, "confidence": 0.9},
                    },
                ],
                "feedback_summary": {
                    "contradicted_count": 1,
                    "recent_feedback": [
                        {
                            "target_type": "action",
                            "target_id": "act-contradicted",
                            "verdict": "contradicted",
                            "applied_status": "contradicted",
                        },
                    ],
                },
                "world_model_lite": {"uncertainty": 0.3, "policy_score": {"cost": 0.1}},
                "memory_health": {"status": "available", "fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=10,
            created_at="2026-01-01T00:00:03+00:00",
        )
        # Contradicted action should rank before unverified episode
        self.assertGreater(len(plan.candidates), 1)
        contradicted_rank = next(
            (i for i, c in enumerate(plan.candidates) if c.target_id == "act-contradicted"),
            None,
        )
        unverified_rank = next(
            (i for i, c in enumerate(plan.candidates) if c.target_id == "ep-unverified"),
            None,
        )
        if contradicted_rank is not None and unverified_rank is not None:
            self.assertLess(contradicted_rank, unverified_rank)

    def test_limit_respected(self) -> None:
        plan = build_replay_plan(
            {
                "runtime_episodes": [
                    {"episode_id": f"ep-{i}", "operation": "query", "status": "succeeded",
                     "created_at": "2026-01-01T00:00:00+00:00",
                     "verification": {"status": "unverified"},
                     "prediction": {"status": "pending"}}
                    for i in range(10)
                ],
                "world_model_lite": {"uncertainty": 0.3},
                "memory_health": {"fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=3,
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertLessEqual(len(plan.candidates), 3 + 2)  # +2 for potential memory/policy candidates

    def test_source_window_bounds_large_history_without_full_scan(self) -> None:
        episodes = _ReplayWindowProbeSequence(
            [
                {
                    "episode_id": f"ep-{i}",
                    "operation": "query",
                    "status": "succeeded",
                    "created_at": f"2026-01-01T00:{(i // 60) % 60:02d}:{i % 60:02d}+00:00",
                    "verification": {"status": "unverified", "confidence": 0.5},
                    "prediction": {"status": "pending", "confidence": 0.5},
                }
                for i in range(500)
            ]
        )

        plan = build_replay_plan(
            {
                "runtime_episode_count": 500,
                "runtime_episodes": episodes,
                "world_model_lite": {"uncertainty": 0.3},
                "memory_health": {"status": "available", "fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=5,
            created_at="2026-01-01T00:00:00+00:00",
        ).to_payload()

        source_window = plan["source_window"]
        self.assertEqual(source_window["surface"], "bounded_replay_plan_source_window.v1")
        self.assertEqual(source_window["window_policy"], REPLAY_PLAN_SOURCE_WINDOW_POLICY)
        self.assertFalse(source_window["runs_live_tick"])
        self.assertEqual(source_window["source_limits"]["runtime_episodes"], REPLAY_PLAN_SOURCE_WINDOW_LIMIT)
        self.assertEqual(source_window["source_counts"]["runtime_episodes"], 500)
        self.assertEqual(source_window["window_counts"]["runtime_episodes"], REPLAY_PLAN_SOURCE_WINDOW_LIMIT)
        self.assertTrue(source_window["truncated_source_counts"]["runtime_episodes"])
        self.assertEqual(episodes.slice_calls, 1)
        self.assertLessEqual(episodes.integer_getitem_calls, 2)

    def test_source_window_uses_newest_first_head_when_timestamps_descend(self) -> None:
        episodes = _ReplayWindowProbeSequence(
            [
                {
                    "episode_id": f"ep-{i}",
                    "operation": "query",
                    "status": "succeeded",
                    "created_at": f"2026-01-02T00:{((500 - i) // 60) % 60:02d}:{(500 - i) % 60:02d}+00:00",
                    "verification": {"status": "unverified", "confidence": 0.5},
                    "prediction": {"status": "pending", "confidence": 0.5},
                }
                for i in range(500)
            ]
        )

        plan = build_replay_plan(
            {
                "runtime_episode_count": 500,
                "runtime_episodes": episodes,
                "world_model_lite": {"uncertainty": 0.3},
                "memory_health": {"status": "available", "fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=10,
            created_at="2026-01-02T00:00:00+00:00",
        ).to_payload()

        candidate_ids = {candidate["target_id"] for candidate in plan["candidates"]}
        self.assertIn("ep-0", candidate_ids)
        self.assertNotIn("ep-499", candidate_ids)
        self.assertEqual(episodes.slice_calls, 1)
        self.assertLessEqual(episodes.integer_getitem_calls, 2)

    def test_feedback_target_stub_preserves_old_high_signal_recall(self) -> None:
        plan = build_replay_plan(
            {
                "runtime_episode_count": 120,
                "runtime_episodes": [
                    {
                        "episode_id": f"ep-{i}",
                        "operation": "query",
                        "status": "succeeded",
                        "created_at": f"2026-01-01T00:{(i // 60) % 60:02d}:{i % 60:02d}+00:00",
                        "verification": {"status": "verified", "confidence": 0.9},
                        "prediction": {"status": "complete", "confidence": 0.9},
                    }
                    for i in range(120)
                ],
                "feedback_summary": {
                    "feedback_count": 41,
                    "contradicted_count": 1,
                    "recent_feedback": [
                        {
                            "feedback_id": "fb-old-ep",
                            "target_type": "runtime_episode",
                            "target_id": "ep-3",
                            "created_at": "2026-01-02T00:00:00+00:00",
                            "verdict": "contradicted",
                            "applied_status": "contradicted",
                            "confidence": 0.95,
                            "summary": "Old episode was contradicted by review.",
                            "has_corrected_output": True,
                        }
                    ]
                    + [
                        {
                            "feedback_id": f"fb-newer-low-signal-{i}",
                            "target_type": "runtime_episode",
                            "target_id": f"ep-{10 + i}",
                            "created_at": f"2026-01-02T00:{(i // 60) % 60:02d}:{(i + 1) % 60:02d}+00:00",
                            "verdict": "unverified",
                            "applied_status": "unverified",
                            "confidence": 0.2,
                            "summary": "Newer low-signal feedback.",
                        }
                        for i in range(40)
                    ],
                },
                "world_model_lite": {"uncertainty": 0.3},
                "memory_health": {"status": "available", "fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=5,
            created_at="2026-01-02T00:00:01+00:00",
        ).to_payload()

        source_window = plan["source_window"]
        self.assertEqual(source_window["source_limits"]["recent_feedback"], REPLAY_PLAN_FEEDBACK_WINDOW_LIMIT)
        self.assertEqual(
            source_window["source_limits"]["feedback_target_stubs"],
            REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
        )
        self.assertTrue(source_window["truncated_source_counts"]["runtime_episodes"])
        self.assertEqual(
            source_window["window_counts"]["feedback_runtime_episode_stubs"],
            REPLAY_PLAN_FEEDBACK_TARGET_STUB_LIMIT,
        )
        self.assertEqual(plan["candidates"][0]["target_id"], "ep-3")
        self.assertEqual(plan["candidates"][0]["operation"], "feedback_target_replay")
        self.assertIn("contradicted_feedback", plan["candidates"][0]["reason_codes"])
        self.assertIn("corrected_output_available", plan["candidates"][0]["reason_codes"])

    def test_runtime_evidence_summary_preserves_source_window(self) -> None:
        plan = build_replay_plan(
            {
                "runtime_episodes": [
                    {
                        "episode_id": "ep-1",
                        "operation": "query",
                        "status": "succeeded",
                        "verification": {"status": "unverified", "confidence": 0.4},
                        "prediction": {"status": "pending", "confidence": 0.4},
                    }
                ],
                "world_model_lite": {"uncertainty": 0.3},
                "memory_health": {"status": "available", "fill_ratio": 0.2},
                "policy_decision": {"action": "continue_current_policy"},
            },
            limit=5,
            created_at="2026-01-01T00:00:00+00:00",
        ).to_payload()

        summary = RuntimeEvidenceReporter()._replay_plan_summary(plan)
        source_window = summary["source_window"]
        self.assertEqual(source_window["surface"], "bounded_replay_plan_source_window.v1")
        self.assertEqual(source_window["window_policy"], REPLAY_PLAN_SOURCE_WINDOW_POLICY)
        self.assertFalse(source_window["runs_live_tick"])
        self.assertEqual(
            source_window["device_placement"]["active_replay_computation"],
            "cpu_service_ranking_only",
        )


class TestReplayConstants(unittest.TestCase):
    """Replay module constants."""

    def test_schema_version(self) -> None:
        self.assertEqual(REPLAY_PLAN_SCHEMA_VERSION, 1)

    def test_priority_rules_version(self) -> None:
        self.assertEqual(REPLAY_PLAN_PRIORITY_RULES_VERSION, "deterministic-v1")

    def test_default_limit(self) -> None:
        self.assertEqual(REPLAY_PLAN_DEFAULT_LIMIT, 20)

    def test_max_limit(self) -> None:
        self.assertEqual(REPLAY_PLAN_MAX_LIMIT, 50)

    def test_safety_boundaries_not_empty(self) -> None:
        self.assertGreater(len(REPLAY_SAMPLE_SAFETY_BOUNDARIES), 0)
        self.assertIn("no_training", REPLAY_SAMPLE_SAFETY_BOUNDARIES)
        self.assertIn("audit_only", REPLAY_SAMPLE_SAFETY_BOUNDARIES)

    def test_priority_weights_keys(self) -> None:
        expected_keys = {
            "safety", "feedback", "uncertainty", "memory_pressure",
            "latency_pressure", "policy_pressure", "provenance_gap", "recency_rank",
        }
        self.assertEqual(set(REPLAY_PLAN_PRIORITY_WEIGHTS.keys()), expected_keys)

    def test_reason_precedence_contradicted_highest(self) -> None:
        self.assertEqual(REPLAY_REASON_PRECEDENCE["contradicted_feedback"], 0)
        self.assertGreater(REPLAY_REASON_PRECEDENCE["healthy_grounded_state"], 0)


class TestDefaultReplaySampleSafetyFlags(unittest.TestCase):
    """_default_replay_sample_safety_flags structure."""

    def test_defaults(self) -> None:
        flags = _default_replay_sample_safety_flags()
        self.assertTrue(flags["audit_only"])
        self.assertFalse(flags["operator_confirmed"])
        self.assertFalse(flags["training_started"])
        self.assertFalse(flags["sleep_started"])
        self.assertFalse(flags["memory_verification_promoted"])
        self.assertTrue(flags["not_promoted"])


class TestCoerceReplaySampleSummary(unittest.TestCase):
    """_coerce_replay_sample_summary round-trip."""

    def test_none_input_gives_defaults(self) -> None:
        result = _coerce_replay_sample_summary(None)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["endpoint"], "/terminus/replay-sample")
        self.assertEqual(result["count"], 0)
        self.assertTrue(result["audit_only"])
        self.assertTrue(result["advisory"])
        self.assertFalse(result["executable"])

    def test_known_values(self) -> None:
        result = _coerce_replay_sample_summary({
            "count": 5,
            "mode_counts": {"execute": 3},
            "status_counts": {"recorded": 5},
            "safety_flags": {"operator_confirmed": True},
        })
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["mode_counts"]["execute"], 3)
        self.assertTrue(result["safety_flags"]["operator_confirmed"])
        self.assertTrue(result["safety_flags"]["audit_only"])  # default preserved


class TestReplayActionForReasons(unittest.TestCase):
    """_replay_action_for_reasons maps reason code sets to actions."""

    def test_contradicted_feedback_gives_review_contradiction(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["contradicted_feedback"]), "review_contradiction")

    def test_failed_runtime_gives_review_contradiction(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["failed_runtime_episode"]), "review_contradiction")

    def test_unverified_feedback_gives_verify_pending(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["unverified_feedback"]), "verify_pending_evidence")

    def test_pending_prediction_gives_verify_pending(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["pending_prediction"]), "verify_pending_evidence")

    def test_memory_pressure_gives_sleep_advisory(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["memory_capacity_pressure"]), "sleep_consolidation_advisory")

    def test_high_latency_gives_reduce_scope(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["high_latency"]), "reduce_scope_or_wait")

    def test_high_uncertainty_gives_collect_evidence(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["high_uncertainty"]), "collect_more_evidence")

    def test_healthy_grounded_gives_continue_observing(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["healthy_grounded_state"]), "continue_observing")

    def test_unknown_reason_gives_replay_episode(self) -> None:
        self.assertEqual(_replay_action_for_reasons(["unknown_reason"]), "replay_episode_for_grounding")


class TestReplayEndpointForAction(unittest.TestCase):
    """_replay_endpoint_for_action maps actions to endpoints."""

    def test_review_contradiction_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("review_contradiction"), "/terminus/runtime-feedback")

    def test_verify_pending_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("verify_pending_evidence"), "/terminus/runtime-feedback")

    def test_sleep_advisory_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("sleep_consolidation_advisory"), "/terminus/living-loop")

    def test_replay_episode_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("replay_episode_for_grounding"), "/terminus/runtime-traces/export")

    def test_reduce_scope_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("reduce_scope_or_wait"), "/terminus")

    def test_unknown_action_endpoint(self) -> None:
        self.assertEqual(_replay_endpoint_for_action("unknown_action"), "/terminus/living-loop")


class TestReplayUniqueReasons(unittest.TestCase):
    """_replay_unique_reasons deduplicates and cleans reason codes."""

    def test_deduplicates(self) -> None:
        self.assertEqual(_replay_unique_reasons(["contradicted", "contradicted"]), ("contradicted",))

    def test_cleans_text(self) -> None:
        self.assertEqual(_replay_unique_reasons(["  CONTRADICTED "]), ("contradicted",))

    def test_removes_empty(self) -> None:
        self.assertEqual(_replay_unique_reasons(["", "valid", "  "]), ("valid",))

    def test_preserves_order(self) -> None:
        self.assertEqual(_replay_unique_reasons(["b", "a", "b"]), ("b", "a"))


class TestReplayReasonPrecedence(unittest.TestCase):
    """_replay_reason_precedence returns the lowest precedence value."""

    def test_known_code(self) -> None:
        self.assertEqual(_replay_reason_precedence(["contradicted_feedback"]), 0)

    def test_unknown_code_gets_99(self) -> None:
        self.assertEqual(_replay_reason_precedence(["unknown_reason"]), 99)

    def test_returns_best_precedence(self) -> None:
        self.assertEqual(_replay_reason_precedence(["healthy_grounded_state", "contradicted_feedback"]), 0)

    def test_empty_returns_99(self) -> None:
        self.assertEqual(_replay_reason_precedence([]), 99)


if __name__ == "__main__":
    unittest.main()
