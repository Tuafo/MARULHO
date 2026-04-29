from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from hecsn.config.model_config import HECSNConfig
from hecsn.cortex.episodic_memory import Provenance
from hecsn.service.api import create_app
from hecsn.service.living_loop import (
    ActionExecutionRecord,
    ConsolidationRecord,
    OperationalSelfModel,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
    WorldModelLiteSummary,
    build_policy_actuator_status,
    build_replay_plan,
    build_runtime_benchmark_telemetry,
    replay_candidate_safety_flags,
)
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_checkpoint(root: Path) -> Path:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)
    return save_trainer_checkpoint(root / "initial.pt", trainer, metadata={"test_case": "living_loop"})


class LivingLoopPrimitiveTests(unittest.TestCase):
    def test_action_execution_record_round_trips_verified_prediction_provenance(self) -> None:
        payload = {
            "action_id": "act-verified",
            "action_type": "workspace_search",
            "inputs": {"query_text": "cats chase mice"},
            "predicted_outcome": "I expect grounded evidence about cats chasing mice.",
            "actual_outcome": "Workspace search found cats chase mice.",
            "verification": {
                "status": "verified",
                "success": True,
                "confidence": 0.91,
                "contradiction": False,
                "summary": "Verified workspace search found matching hits.",
                "evidence": [{"path": "notes.md", "line_number": 2}],
            },
            "topics": ["cats", "mice"],
            "recorded_at": "2026-01-01T00:00:00+00:00",
        }

        record = ActionExecutionRecord.from_payload(payload)
        round_tripped = ActionExecutionRecord.from_payload(record.to_payload())

        self.assertEqual(record.prediction.status, PredictionStatus.FULFILLED)
        self.assertEqual(record.prediction.provenance, Provenance.VERIFIED)
        self.assertEqual(record.verification.status, VerificationStatus.VERIFIED)
        self.assertEqual(record.prediction.prediction_id, round_tripped.prediction.prediction_id)
        self.assertEqual(record.verification.verification_id, round_tripped.verification.verification_id)
        self.assertEqual(round_tripped.verification.evidence[0]["path"], "notes.md")

    def test_action_execution_record_preserves_contradicted_provenance(self) -> None:
        payload = {
            "action_id": "act-contradicted",
            "action_type": "workspace_read",
            "predicted_outcome": "I expect missing.md to exist.",
            "actual_outcome": "Could not open missing.md.",
            "verification": {
                "status": "contradicted",
                "success": False,
                "confidence": 0.74,
                "contradiction": True,
                "summary": "The read contradicted the expected file existence.",
            },
        }

        record = ActionExecutionRecord.from_payload(payload)

        self.assertEqual(record.prediction.status, PredictionStatus.CONTRADICTED)
        self.assertEqual(record.prediction.provenance, Provenance.CONTRADICTED)
        self.assertEqual(record.verification.status, VerificationStatus.CONTRADICTED)

    def test_provenance_state_exposes_each_typed_bucket(self) -> None:
        state = ProvenanceState.from_distribution(
            {
                "observed": 2,
                "inferred": 3,
                "dreamed": 4,
                "verified": 5,
                "contradicted": 6,
            }
        )

        payload = state.to_payload()

        self.assertEqual(payload["observed"], 2)
        self.assertEqual(payload["inferred"], 3)
        self.assertEqual(payload["dreamed"], 4)
        self.assertEqual(payload["verified"], 5)
        self.assertEqual(payload["contradicted"], 6)
        self.assertEqual(payload["total"], 20)

    def test_world_model_lite_scores_verified_contradicted_and_pending_predictions(self) -> None:
        action_payloads = [
            {
                "action_id": "act-verified",
                "action_type": "workspace_search",
                "predicted_outcome": "I expect grounded evidence.",
                "actual_outcome": "Found grounded evidence.",
                "verification": {
                    "status": "verified",
                    "success": True,
                    "confidence": 0.9,
                    "contradiction": False,
                    "evidence": [{"path": "notes.md"}],
                },
            },
            {
                "action_id": "act-contradicted",
                "action_type": "workspace_read",
                "predicted_outcome": "I expect missing.md to exist.",
                "actual_outcome": "missing.md was absent.",
                "verification": {
                    "status": "contradicted",
                    "success": False,
                    "confidence": 0.8,
                    "contradiction": True,
                },
            },
            {
                "action_id": "act-pending",
                "action_type": "workspace_search",
                "predicted_outcome": "I expect later evidence.",
                "verification": {
                    "status": "unverified",
                    "success": False,
                    "confidence": 0.2,
                    "contradiction": False,
                },
            },
        ]
        actions = tuple(ActionExecutionRecord.from_payload(payload) for payload in action_payloads)
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=1,
            configured=True,
            running=False,
            provenance=ProvenanceState(),
            predictions=[item.prediction for item in actions],
            actions=actions,
            action_loop={"actions_recorded": len(actions)},
        )

        summary = model.to_payload()["world_model_lite"]

        self.assertEqual(summary["prediction_count"], 3)
        self.assertEqual(summary["fulfilled_count"], 1)
        self.assertEqual(summary["contradicted_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertAlmostEqual(summary["prediction_accuracy"], 0.5)
        self.assertAlmostEqual(summary["contradiction_rate"], 1 / 3)
        self.assertEqual(summary["verification_count"], 3)
        self.assertEqual(summary["verified_action_count"], 1)
        self.assertEqual(summary["contradicted_action_count"], 1)
        self.assertEqual(summary["unverified_action_count"], 1)
        self.assertAlmostEqual(summary["verification_success_rate"], 1 / 3)
        self.assertEqual(summary["recommended_next_action"], "investigate_contradictions")
        self.assertEqual(
            summary["policy_score"]["recommended_next_action"],
            summary["recommended_next_action"],
        )
        self.assertGreater(summary["information_gain"], 0.0)
        self.assertGreater(summary["risk"], 0.0)
        self.assertIn("confidence_uncertainty", summary["components"])

    def test_skill_memory_records_are_derived_from_action_history(self) -> None:
        actions = tuple(
            ActionExecutionRecord.from_payload(payload)
            for payload in [
                {
                    "action_id": "act-search-ok",
                    "action_type": "workspace_search",
                    "inputs": {"query_text": "cats"},
                    "predicted_outcome": "Find cat notes.",
                    "actual_outcome": "Found cat notes.",
                    "verification": {"status": "verified", "success": True, "confidence": 0.9},
                    "topics": ["cats"],
                    "recorded_at": "2026-01-02T00:00:00+00:00",
                    "trigger_reason": "autonomy",
                    "trigger_query_text": "cats",
                },
                {
                    "action_id": "act-search-fail",
                    "action_type": "workspace_search",
                    "inputs": {"query_text": "dogs"},
                    "predicted_outcome": "Find dog notes.",
                    "actual_outcome": "No dog notes found.",
                    "verification": {
                        "status": "contradicted",
                        "success": False,
                        "confidence": 0.8,
                        "contradiction": True,
                        "summary": "Expected dog notes were absent.",
                    },
                    "topics": ["dogs"],
                    "recorded_at": "2026-01-03T00:00:00+00:00",
                    "trigger_reason": "operator",
                    "trigger_query_text": "dogs",
                },
            ]
        )

        memories = SkillMemoryRecord.from_action_records(actions)
        search = memories[0]

        self.assertEqual(search.action_type, "workspace_search")
        self.assertEqual(search.tool, "workspace_search")
        self.assertEqual(search.action_count, 2)
        self.assertEqual(search.success_count, 1)
        self.assertEqual(search.failure_count, 1)
        self.assertAlmostEqual(search.success_rate, 0.5)
        self.assertEqual(search.status, "mixed")
        self.assertEqual(search.last_used_at, "2026-01-03T00:00:00+00:00")
        self.assertEqual(search.preconditions["observed_input_keys"], ["query_text"])
        self.assertEqual(search.trigger_context["trigger_reasons"], ["autonomy", "operator"])
        self.assertIn("Find cat notes.", search.expected_outcomes)
        self.assertEqual(search.failure_modes[0]["action_id"], "act-search-fail")
        self.assertEqual(set(search.topics), {"cats", "dogs"})

    def test_consolidation_payload_exposes_semantic_state_flags_without_replacing_status(self) -> None:
        record = ConsolidationRecord.from_payload(
            {
                "record_id": "delayed-1",
                "query_text": "cats chase mice",
                "query_terms": ["cats", "mice"],
                "source_weights": {"notes.md": 1.0},
                "provider_weights": {"workspace": 1.0},
                "credit_events": 1,
                "penalty_events": 1,
                "aggregate_count": 2,
                "aggregation_events": 1,
                "trajectory_net_score": 0.25,
            }
        )

        payload = record.to_payload()

        self.assertEqual(payload["status"], "credited")
        self.assertIn("observed", payload["consolidation_states"])
        self.assertIn("summarized", payload["consolidation_states"])
        self.assertIn("verified", payload["consolidation_states"])
        self.assertIn("contradicted", payload["consolidation_states"])
        self.assertIn("semanticized", payload["consolidation_states"])
        self.assertEqual(payload["semantic_terms"], ["cats", "mice"])

    def test_runtime_benchmark_telemetry_summarizes_synthetic_runtime_samples(self) -> None:
        episodes = tuple(
            RuntimeEpisodeTrace.from_payload(payload)
            for payload in [
                {
                    "episode_id": "feed-1",
                    "operation": "feed",
                    "status": "succeeded",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "latency_ms": 100.0,
                    "actual_output": {"tokens_processed": 50},
                    "verification": {"status": "verified", "success": True},
                },
                {
                    "episode_id": "query-1",
                    "operation": "query",
                    "status": "succeeded",
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "latency_ms": 200.0,
                    "verification": {"status": "verified", "success": True},
                },
                {
                    "episode_id": "respond-1",
                    "operation": "respond",
                    "status": "failed",
                    "created_at": "2026-01-01T00:00:02+00:00",
                    "latency_ms": 300.0,
                    "failure": {"message": "synthetic failure"},
                    "verification": {"status": "contradicted", "success": False, "contradiction": True},
                },
                {
                    "episode_id": "action-episode-1",
                    "operation": "runtime_action",
                    "status": "succeeded",
                    "created_at": "2026-01-01T00:00:03+00:00",
                    "latency_ms": 80.0,
                    "verification": {"status": "verified", "success": True},
                },
            ]
        )
        actions = tuple(
            ActionExecutionRecord.from_payload(payload)
            for payload in [
                {
                    "action_id": "act-ok",
                    "action_type": "workspace_search",
                    "predicted_outcome": "Find evidence.",
                    "verification": {"status": "verified", "success": True, "confidence": 0.8},
                },
                {
                    "action_id": "act-bad",
                    "action_type": "workspace_read",
                    "predicted_outcome": "Read missing evidence.",
                    "execution_status": "failed",
                    "verification": {"status": "contradicted", "success": False, "contradiction": True},
                },
            ]
        )
        world = WorldModelLiteSummary.from_records(actions=actions)

        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
            action_loop={"actions_recorded": 2},
            memory={"size": 4, "capacity": 16, "fill_ratio": 0.25, "total_stored": 4, "total_evicted": 0},
            cortex={
                "enabled": True,
                "thoughts_generated": 2,
                "dreams_generated": 1,
                "episodic_memory": {
                    "embedder": {
                        "kind": "NIMEmbedder",
                        "nim_calls": 4,
                        "rate_limit_hits": 1,
                        "cache_hits": 3,
                        "cache_misses": 1,
                        "cache_size": 3,
                    }
                },
            },
            replay_sample_summary={
                "count": 2,
                "history_count": 2,
                "selected_count": 3,
                "latest_selected_count": 1,
                "mode_counts": {"sample": 1, "execute": 1},
                "status_counts": {"recorded": 2},
                "latest_history_item": {
                    "replay_sample_id": "replay-execute-1",
                    "mode": "execute",
                    "status": "recorded",
                    "selected_count": 1,
                    "safety_flags": {"audit_only": True, "external_calls_made": False},
                },
                "safety_flags": {"audit_only": True, "external_calls_made": False},
            },
        )

        self.assertEqual(telemetry["endpoint_latency_ms"]["feed"]["count"], 1)
        self.assertEqual(telemetry["endpoint_latency_ms"]["query"]["avg_ms"], 200.0)
        self.assertEqual(telemetry["endpoint_latency_ms"]["respond"]["failure_count"], 1)
        self.assertEqual(telemetry["endpoint_latency_ms"]["runtime_action"]["latency_count"], 1)
        self.assertEqual(telemetry["endpoint_latency_ms"]["runtime_action"]["count"], 3)
        self.assertAlmostEqual(telemetry["tokens_per_second"]["value"], 500.0)
        self.assertEqual(telemetry["memory"]["status"], "available")
        self.assertEqual(telemetry["nim"]["observed_call_count"], 7)
        self.assertIsNone(telemetry["nim"]["calls_per_minute"])
        self.assertEqual(telemetry["nim"]["rate_limit_hits"], 1)
        self.assertAlmostEqual(telemetry["cache"]["hit_rate"], 0.75)
        self.assertAlmostEqual(telemetry["action_success"]["success_rate"], 0.5)
        self.assertAlmostEqual(telemetry["verification_success"]["success_rate"], 0.5)
        self.assertEqual(telemetry["policy_recommendations"]["counts"]["investigate_contradictions"], 1)
        self.assertEqual(telemetry["replay_sample_summary"]["count"], 2)
        self.assertEqual(telemetry["replay_sample_summary"]["mode_counts"]["execute"], 1)
        self.assertEqual(telemetry["replay_sample_summary"]["status_counts"]["recorded"], 2)
        self.assertEqual(telemetry["replay_sample_summary"]["latest_history_item"]["replay_sample_id"], "replay-execute-1")
        self.assertTrue(telemetry["replay_sample_summary"]["safety_flags"]["audit_only"])
        self.assertFalse(telemetry["replay_sample_summary"]["safety_flags"]["external_calls_made"])

    def test_operational_self_model_exposes_derived_self_model_surfaces(self) -> None:
        actions = tuple(
            ActionExecutionRecord.from_payload(payload)
            for payload in [
                {
                    "action_id": "act-verified",
                    "action_type": "workspace_search",
                    "inputs": {"query_text": "cats"},
                    "predicted_outcome": "Find cat evidence.",
                    "actual_outcome": "Found cat evidence.",
                    "verification": {
                        "status": "verified",
                        "success": True,
                        "confidence": 0.9,
                        "evidence": [{"path": "notes.md"}],
                    },
                    "topics": ["cats"],
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                },
                {
                    "action_id": "act-contradicted",
                    "action_type": "workspace_read",
                    "inputs": {"path": "missing.md"},
                    "predicted_outcome": "Read missing file.",
                    "actual_outcome": "missing.md was absent.",
                    "verification": {
                        "status": "contradicted",
                        "success": False,
                        "confidence": 0.7,
                        "contradiction": True,
                        "summary": "File was absent.",
                    },
                    "topics": ["missing"],
                    "recorded_at": "2026-01-02T00:00:00+00:00",
                },
            ]
        )
        model = OperationalSelfModel.build(
            token_count=12,
            state_revision=3,
            configured=True,
            running=True,
            provenance=ProvenanceState(verified=1, contradicted=1, total=2),
            predictions=[item.prediction for item in actions],
            actions=actions,
            action_loop={
                "actions_recorded": 10,
                "supported_actions": ["workspace_search", "workspace_read", "web_fetch"],
            },
            memory={
                "size": 4,
                "capacity": 16,
                "fill_ratio": 0.25,
                "total_stored": 4,
                "total_evicted": 0,
                "provenance_distribution": {"verified": 1, "contradicted": 1},
            },
            cortex={"enabled": True, "memory_count": 4},
        )

        payload = model.to_payload()

        self.assertIn("digital_action_execution", payload["capabilities"])
        self.assertEqual(payload["limits"]["action_history_count"], 10)
        self.assertTrue(payload["limits"]["action_history_truncated"])
        self.assertEqual(payload["budgets"]["memory_size"], 4)
        self.assertEqual(payload["memory_health"]["status"], "available")
        self.assertEqual(payload["grounding_health"]["status"], "contradictions_present")
        self.assertEqual(payload["recent_failures"][0]["action_id"], "act-contradicted")
        self.assertEqual(payload["uncertain_domains"][0]["domain"], "missing")
        self.assertEqual({item["name"] for item in payload["tools"]}, {"workspace_search", "workspace_read", "web_fetch"})
        skill_by_tool = {item["tool"]: item for item in payload["skill_memories"]}
        self.assertEqual(skill_by_tool["workspace_search"]["status"], "verified")
        self.assertEqual(skill_by_tool["workspace_read"]["status"], "contradicted")

    def test_policy_actuator_prioritizes_contradictions_over_memory_and_latency_pressure(self) -> None:
        policy = build_policy_actuator_status(
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
                            "target_id": "act-contradicted",
                            "verdict": "contradicted",
                            "applied_status": "contradicted",
                        }
                    ],
                },
                "actions": [
                    {
                        "action_id": "act-contradicted",
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
        ).to_payload()

        self.assertEqual(policy["action"], "investigate_contradictions")
        self.assertTrue(policy["advisory"])
        self.assertFalse(policy["executable"])
        self.assertEqual(policy["target_action_id"], "act-contradicted")
        self.assertIn("contradicted_feedback", {reason["code"] for reason in policy["reasons"]})

    def test_policy_actuator_prioritizes_pending_evidence_before_sleep_pressure(self) -> None:
        policy = build_policy_actuator_status(
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
        ).to_payload()

        self.assertEqual(policy["action"], "verify_pending_evidence")
        self.assertEqual(policy["target_action_id"], "act-pending")
        self.assertIn("pending_predictions", {reason["code"] for reason in policy["reasons"]})

    def test_policy_actuator_reduces_scope_before_collecting_uncertain_evidence(self) -> None:
        policy = build_policy_actuator_status(
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
        ).to_payload()

        self.assertEqual(policy["action"], "reduce_scope_or_wait")
        self.assertIn("high_latency", {reason["code"] for reason in policy["reasons"]})

    def test_replay_plan_prioritizes_contradicted_feedback_and_is_advisory(self) -> None:
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
                            }
                        ],
                    }
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
                        }
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

        self.assertEqual(plan["schema_version"], 1)
        self.assertTrue(plan["advisory"])
        self.assertFalse(plan["executable"])
        self.assertEqual(plan["endpoint"], "/terminus/replay-plan")
        self.assertEqual(plan["priority_rules_version"], "deterministic-v1")
        self.assertEqual(plan["snapshot_counts"]["runtime_episodes"], 1)
        self.assertGreaterEqual(plan["count"], 1)
        top = plan["candidates"][0]
        self.assertEqual(top["target_type"], "runtime_episode")
        self.assertEqual(top["target_id"], "episode-contradicted")
        self.assertEqual(top["suggested_consolidation_action"], "review_contradiction")
        self.assertEqual(top["suggested_endpoint"], "/terminus/runtime-feedback")
        self.assertIn("contradicted_feedback", top["reason_codes"])
        self.assertIn("corrected_output_available", top["reason_codes"])
        self.assertIn("high_latency", top["reason_codes"])
        self.assertGreater(top["priority_score"], 100.0)
        self.assertEqual(top["feedback"]["contradicted_count"], 1)
        self.assertTrue(top["suggested_input"]["operator_review_required"])

    def test_replay_candidate_safety_flags_keep_contradictions_non_factual(self) -> None:
        flags = replay_candidate_safety_flags(
            {
                "candidate_id": "replay-contradicted",
                "reason_codes": ["contradicted_feedback"],
                "suggested_consolidation_action": "review_contradiction",
                "feedback": {"contradicted_count": 1},
                "provenance": {"provenance": "synthetic_dream"},
            }
        )

        self.assertTrue(flags["audit_only"])
        self.assertTrue(flags["not_promoted"])
        self.assertFalse(flags["promoted_to_verified_fact"])
        self.assertTrue(flags["non_factual"])
        self.assertTrue(flags["negative_lesson"])
        self.assertTrue(flags["dreamed_or_synthetic"])
        self.assertIn("no_training", flags["safety_boundaries"])

    def test_service_status_and_endpoint_include_derived_living_loop_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(_build_checkpoint(root), trace_dir=root / "traces", env_root=root)
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    },
                )
                status_response = client.get("/status")
                living_loop_response = client.get("/terminus/living-loop")

        self.assertEqual(action_response.status_code, 200)
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(living_loop_response.status_code, 200)
        living_loop = status_response.json()["terminus_runtime"]["living_loop"]
        endpoint_loop = living_loop_response.json()["living_loop"]
        self.assertEqual(living_loop["action_count"], 1)
        self.assertEqual(living_loop["prediction_count"], 1)
        self.assertEqual(living_loop["predictions"][0]["status"], "fulfilled")
        self.assertEqual(living_loop["predictions"][0]["provenance"], "verified")
        world_model_lite = living_loop["world_model_lite"]
        self.assertEqual(world_model_lite["prediction_count"], 1)
        self.assertEqual(world_model_lite["fulfilled_count"], 1)
        self.assertEqual(world_model_lite["contradicted_count"], 0)
        self.assertAlmostEqual(world_model_lite["prediction_accuracy"], 1.0)
        self.assertEqual(
            world_model_lite["policy_score"]["recommended_next_action"],
            world_model_lite["recommended_next_action"],
        )
        self.assertEqual(living_loop["policy_decision"]["schema_version"], 1)
        self.assertEqual(living_loop["policy_decision"]["action"], "continue_current_policy")
        self.assertTrue(living_loop["policy_decision"]["advisory"])
        self.assertFalse(living_loop["policy_decision"]["executable"])
        self.assertEqual(living_loop["replay_plan"]["schema_version"], 1)
        self.assertTrue(living_loop["replay_plan"]["advisory"])
        self.assertFalse(living_loop["replay_plan"]["executable"])
        self.assertEqual(living_loop["replay_plan"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(endpoint_loop["world_model_lite"]["fulfilled_count"], 1)
        self.assertEqual(endpoint_loop["actions"][0]["verification"]["status"], "verified")
        required_living_loop_fields = {
            "prediction_count",
            "action_count",
            "world_model_lite",
            "skill_memories",
            "capabilities",
            "limits",
            "budgets",
            "memory_health",
            "grounding_health",
            "benchmark_telemetry",
            "replay_plan",
        }
        self.assertLessEqual(required_living_loop_fields, set(living_loop))
        self.assertLessEqual(required_living_loop_fields, set(endpoint_loop))
        self.assertEqual(endpoint_loop["prediction_count"], 1)
        self.assertEqual(endpoint_loop["action_count"], 1)
        self.assertIn("digital_action_execution", endpoint_loop["capabilities"])
        self.assertEqual(endpoint_loop["limits"]["snapshot_action_count"], 1)
        self.assertEqual(endpoint_loop["budgets"]["action_snapshot_used"], 1)
        self.assertEqual(endpoint_loop["memory_health"]["status"], "no_memory_snapshot")
        self.assertEqual(endpoint_loop["grounding_health"]["status"], "grounded")
        self.assertIn("drives", endpoint_loop["cortex"])
        benchmark = living_loop["benchmark_telemetry"]
        endpoint_benchmark = endpoint_loop["benchmark_telemetry"]
        self.assertEqual(benchmark["sample"]["action_count"], 1)
        self.assertEqual(benchmark["endpoint_latency_ms"]["runtime_action"]["count"], 1)
        self.assertAlmostEqual(benchmark["action_success"]["success_rate"], 1.0)
        self.assertAlmostEqual(benchmark["verification_success"]["success_rate"], 1.0)
        self.assertEqual(benchmark["memory"]["status"], "available")
        self.assertEqual(endpoint_benchmark["sample"]["action_count"], 1)
        self.assertIn(
            endpoint_benchmark["policy_recommendations"]["latest"],
            endpoint_benchmark["policy_recommendations"]["counts"],
        )
        self.assertEqual(endpoint_benchmark["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(set(endpoint_loop["provenance"]["distribution"].keys()), {
            "observed",
            "inferred",
            "dreamed",
            "verified",
            "contradicted",
        })

    def test_feed_query_and_respond_capture_runtime_episode_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = HECSNServiceManager(_build_checkpoint(root), trace_dir=root / "traces", env_root=root)
            try:
                feed_result = manager.feed(text="Cats chase mice at night. Cats rest indoors during the day. " * 4)
                query_result = manager.query(query_text="cats chase mice", top_k_memories=4)
                respond_result = manager.respond(
                    query_text="cats chase mice",
                    top_k_memories=4,
                    learn_mode="none",
                )
                feedback_result = manager.record_runtime_feedback(
                    {
                        "target_type": "runtime_episode",
                        "target_id": respond_result["runtime_episode"]["episode_id"],
                        "verdict": "contradicted",
                        "confidence": 0.91,
                        "summary": "Manual review found the response overclaimed evidence.",
                        "corrected_output": {
                            "response_text": "Cats chase mice at night.",
                            "password": "secret-value",
                        },
                        "evidence": [{"note": "manual review", "api_key": "secret-value"}],
                        "tags": ["Manual", "Runtime"],
                        "evaluator_id": "qa-1",
                    }
                )
                living_loop = manager.living_loop_status()["living_loop"]
                with manager._lock:
                    latest = manager._runtime_episode_traces[0]
                    latest.setdefault("request", {})["raw_environment"] = {"NVIDIA_API_KEY": "secret-value"}
                    latest.setdefault("request", {})["token_count"] = 123
                    latest.setdefault("prediction", {})["api_key"] = "secret-value"
                    latest.setdefault("actual_output", {})["dotenv_path"] = str(root / ".env")
                    latest.setdefault("verification", {})["password"] = "secret-value"
                trace_export = manager.export_runtime_trace_examples(limit=2)
                respond_export = manager.export_runtime_trace_examples(limit=5, endpoint="respond")
                trace_paths_exist = (
                    Path(feed_result["runtime_episode"]["trace_path"]).exists()
                    and Path(query_result["runtime_episode"]["trace_path"]).exists()
                    and Path(respond_result["runtime_episode"]["trace_path"]).exists()
                )
            finally:
                manager.close()

        feed_episode = feed_result["runtime_episode"]
        query_episode = query_result["runtime_episode"]
        respond_episode = respond_result["runtime_episode"]

        self.assertEqual(feed_episode["operation"], "feed")
        self.assertEqual(query_episode["operation"], "query")
        self.assertEqual(respond_episode["operation"], "respond")
        self.assertEqual(feed_episode["verification"]["status"], "verified")
        self.assertEqual(query_episode["verification"]["status"], "verified")
        self.assertIn(respond_episode["verification"]["status"], {"verified", "unverified"})
        self.assertGreaterEqual(float(feed_episode["latency_ms"]), 0.0)
        self.assertIn("predicted_output", feed_episode["prediction"])
        self.assertIn("action_type", query_episode["action"])
        self.assertIn("response_text", respond_episode["actual_output"])
        self.assertTrue(trace_paths_exist)

        operations = [episode["operation"] for episode in living_loop["runtime_episodes"]]
        self.assertLessEqual({"feed", "query", "respond"}, set(operations))
        self.assertGreaterEqual(living_loop["runtime_episode_count"], 3)
        self.assertGreaterEqual(living_loop["prediction_count"], 3)
        self.assertIn("runtime_episode_trace", living_loop["capabilities"])
        self.assertTrue(feedback_result["accepted"])
        self.assertEqual(living_loop["feedback_count"], 1)
        self.assertEqual(living_loop["feedback_summary"]["contradicted_count"], 1)
        self.assertEqual(living_loop["feedback_summary"]["target_counts"]["runtime_episode"], 1)
        self.assertEqual(living_loop["recent_feedback"][0]["target_type"], "runtime_episode")
        self.assertEqual(living_loop["recent_feedback"][0]["verdict"], "contradicted")
        self.assertEqual(living_loop["grounding_health"]["feedback_count"], 1)
        self.assertEqual(living_loop["grounding_health"]["feedback_impact"], "contradictions_present")
        self.assertEqual(living_loop["benchmark_telemetry"]["feedback"]["feedback_count"], 1)
        self.assertEqual(living_loop["benchmark_telemetry"]["feedback"]["contradicted_count"], 1)
        self.assertEqual(living_loop["policy_decision"]["schema_version"], 1)
        self.assertEqual(living_loop["policy_decision"]["action"], "investigate_contradictions")
        self.assertIn(
            "contradicted_feedback",
            {reason["code"] for reason in living_loop["policy_decision"]["reasons"]},
        )
        self.assertEqual(trace_export["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertIn("not_training", trace_export["training_role"])
        self.assertEqual(trace_export["count"], 2)
        self.assertEqual(trace_export["policy_decision"]["action"], "investigate_contradictions")
        self.assertEqual(trace_export["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertIn("contradicted_feedback", trace_export["replay_plan_summary"]["plan_reason_codes"])
        self.assertEqual(len(trace_export["examples"]), 2)
        exported_example = trace_export["examples"][0]
        self.assertEqual(exported_example["endpoint"], "/respond")
        self.assertEqual(exported_example["type"], "respond")
        self.assertIn("context", exported_example)
        self.assertIn("prediction", exported_example)
        self.assertIn("actual_output", exported_example)
        self.assertIn("verification", exported_example)
        self.assertIn("feedback", exported_example)
        self.assertIn("feedback_summary", exported_example)
        self.assertEqual(exported_example["feedback_summary"]["feedback_count"], 1)
        self.assertEqual(exported_example["feedback_summary"]["contradicted_count"], 1)
        self.assertEqual(exported_example["feedback"][0]["evidence"][0]["note"], "manual review")
        self.assertEqual(exported_example["policy_decision"]["action"], "investigate_contradictions")
        self.assertIn("contradicted_feedback", exported_example["policy_decision"]["reason_codes"])
        self.assertEqual(exported_example["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertIn("provenance", exported_example)
        self.assertIn("latency_ms", exported_example)
        self.assertIsNotNone(exported_example["state_revision"])
        self.assertIsNotNone(exported_example["token_count"])
        self.assertEqual(respond_export["count"], 1)
        self.assertEqual(respond_export["examples"][0]["operation"], "respond")
        self.assertEqual(respond_export["examples"][0]["feedback_summary"]["feedback_count"], 1)
        self.assertIn("raw_environment", trace_export["excluded_fields"])
        export_json = json.dumps(trace_export["examples"], sort_keys=True)
        self.assertNotIn("secret-value", export_json)
        self.assertNotIn("raw_environment", export_json)
        self.assertNotIn("NVIDIA_API_KEY", export_json)
        self.assertNotIn("dotenv_path", export_json)
        self.assertNotIn("api_key", export_json)
        self.assertNotIn("password", export_json)

    def test_tutorial_documents_living_loop_endpoint_latency_posture_and_arc_boundary(self) -> None:
        tutorial = (Path(__file__).resolve().parents[1] / "TERMINUS_Tutorial.md").read_text(encoding="utf-8")

        self.assertIn(
            "observe -> predict -> error/salience/drives -> reason/act -> verify -> typed memory "
            "-> replay/consolidation -> self-model update",
            tutorial,
        )
        self.assertIn("/terminus/living-loop", tutorial)
        for field in (
            "prediction_count",
            "action_count",
            "world_model_lite",
            "skill_memories",
            "capabilities",
            "limits",
            "budgets",
            "memory_health",
            "grounding_health",
            "feedback_summary",
            "feedback_count",
            "verified_feedback_count",
            "contradicted_feedback_count",
            "unverified_feedback_count",
            "recent_feedback",
            "replay_plan",
        ):
            self.assertIn(field, tutorial)
        for feedback_posture in (
            "POST /terminus/runtime-feedback",
            '"target_type": "runtime_episode"',
            '"target_type": "action"',
            "`corrected_output`",
            "`verified` means the target survived review",
            "`contradicted` means the target was wrong",
            "`unverified` means the target is not accepted as grounded yet",
            "`feedback_status`",
            "`feedback_provenance`",
            "`verification.last_feedback_id`",
            "`status=contradictions_present`",
            "`needs_verification`",
            "sanitized per-example `feedback` and `feedback_summary`",
            "`benchmark_telemetry.feedback`",
            "`living_loop_benchmark_telemetry.feedback`",
            "service-benchmark `feedback_telemetry`",
            "`grounding_impact`",
            "make the next self-model snapshot reflect what was actually verified, "
            "contradicted, or still unresolved",
        ):
            self.assertIn(feedback_posture, tutorial)
        for policy_actuator_posture in (
            "GET /terminus/policy-actuator",
            "`policy_decision`",
            "`investigate_contradictions`",
            "`verify_pending_evidence`",
            "`consolidate_or_sleep`",
            "`reduce_scope_or_wait`",
            "`collect_more_evidence`",
            "`continue_current_policy`",
            "Recommendation priority is intentionally conservative",
            "Contradictions first",
            "Pending evidence next",
            "Maintenance pressure before new evidence",
            "Cost/latency pressure before exploration",
            "Uncertainty after safety/cost checks",
            "The safety boundary is strict",
            "**does not execute actions, mutate action history, advance state revision, start sleep, "
            "post feedback, call the cortex, or change runtime configuration**",
            "`suggested_endpoint` and `suggested_input` are operator guidance",
            "`benchmark_telemetry.policy_recommendations` includes `total`, `latest`, `counts`",
            "sanitized top-level `policy_decision`",
            "each exported example also includes sanitized `policy_decision`",
            "records it in `endpoint_timings` / `endpoints_by_name.policy_actuator`",
            "`policy_actuator_summary`",
            "the next living-loop step",
        ):
            self.assertIn(policy_actuator_posture, tutorial)
        for replay_posture in (
            "GET /terminus/replay-plan",
            "`priority_rules_version=deterministic-v1`",
            "`priority_weights`",
            "`plan_reason_codes`",
            "`snapshot_counts`",
            "`candidates`",
            "`target_type` can be `runtime_episode`, `action`, `prediction`, `feedback`, `memory_health`, `uncertain_domain`, or `policy_decision`",
            "`contradicted_feedback`",
            "`corrected_output_available`",
            "`memory_capacity_pressure`",
            "`fatigue_sleep_pressure`",
            "`review_contradiction`",
            "`verify_pending_evidence`",
            "`sleep_consolidation_advisory`",
            "**does not start sleep, replay memories, train adapters, mutate runtime state, post feedback, or execute actions**",
            "complementary learning systems",
            "prioritized experience replay",
            "P(i) proportional to (epsilon + score)^alpha",
            "`benchmark_telemetry.replay_plan_summary`",
            "sanitized top-level `replay_plan_summary`",
            "each exported example also includes sanitized `replay_plan_summary`",
            "records it in `endpoint_timings` / `endpoints_by_name.replay_plan`",
            "`replay_plan_summary`",
            "Dreamed/imagination memories must remain provenance-tagged",
            "POST /terminus/replay-sample",
            "POST /terminus/replay-execute",
            "GET /terminus/replay-sample/history",
            "GET /terminus/replay-execute/history",
            "`mode` — `dry_run`, `sample`, or `execute`",
            "`operator_id` — required non-empty operator identifier",
            "`confirmation` — must be `true`",
            "`alpha` — PER-style priority exponent",
            "`seed` — optional deterministic seed",
            "`replay_sample_id`",
            "`execution_id`",
            "`selected_candidates`",
            "`safety_checks`",
            "`safety_flags`",
            "`before` / `after`",
            "`plan_summary`",
            "`training_started=false`",
            "`sleep_started=false`",
            "`memory_verification_promoted=false`",
            "`feedback_posted=false`",
            "`digital_action_executed=false`",
            "`external_calls_made=false`",
            "`living_loop.replay_sample_summary`",
            "`living_loop.replay_executor_summary`",
            "`benchmark_telemetry.replay_sample_summary`",
            "`benchmark_telemetry.replay_executor_summary`",
            "sanitized top-level `replay_sample_summary`",
            "sanitized top-level `replay_sample_summary` and `replay_executor_summary`",
            "each exported example includes sanitized `replay_sample_summary`",
            "`endpoints_by_name.replay_sample_history`",
            "`replay_sample_summary` and `replay_executor_summary`",
            "operator-gated audit/sample only",
            "Contradicted candidates are negative lessons",
            "Dreamed, synthetic, simulated, contradicted, or failed candidates remain provenance-tagged",
            "the current dashboard remains read-only for replay",
            "does not post to `/terminus/replay-sample` or `/terminus/replay-execute`",
            "Remaining work before autonomous replay learning",
            "GET /terminus/replay-dataset/preview",
            "GET /terminus/replay-dataset/candidates",
            "GET /terminus/replay-dataset/history",
            "python -m hecsn.service.replay_dataset_runner",
            "`training_role=replay_dataset_preview_only_not_training_no_mutation`",
            "`positive_count` and `negative_count`",
            "`provenance_counts` and `example_type_counts`",
            "`replay_dataset_summary`",
            "`replay_dataset_candidates_summary`",
            "`replay_dataset_history_summary`",
            "Curated replay dataset card",
            "preview/export artifacts only",
            "They do not train adapters, rewrite or promote memories, post feedback, execute digital actions, call external tools, start sleep",
        ):
            self.assertIn(replay_posture, tutorial)
        for latency_posture in (
            "There is **no separate fast query API**",
            "semantic query-term matching uses bounded in-request caches",
            "allow_sleep_maintenance=False",
            "sleep_maintenance_deferred",
            "Background/runtime trainer behavior remains unchanged",
            "service construction plus `/health` and `/status` avoid eager NIM/embedder calls",
            "benchmark_telemetry",
            "endpoint_latency_ms",
            "tokens_per_second",
            "policy_recommendations",
            "python -m hecsn.service.trace_export_runner",
            "terminus_runtime_trace_dataset_preview",
            "adapter_distillation_dataset_preview_only_not_training",
            "excluded_fields",
            "checkpoint_contains_no_persisted_runtime_episode_traces",
            "python -m hecsn.evaluation.service_benchmark",
            "hecsn_service_endpoint_latency",
            "endpoint_timings",
            "trace_export_summary",
        ):
            self.assertIn(latency_posture, tutorial)
        for arc_requirement in (
            "ARC-AGI should remain a **separate benchmark path**",
            "object parser",
            "tiny deterministic DSL/search scaffold",
            "benchmark plumbing for ARC experiments",
            "not core living-loop evidence",
            "DSL/program synthesis",
            "verifier",
            "search/refinement",
            "LLM candidates",
            "exact-match scoring",
            "does **not** imply that Terminus already solves ARC-AGI",
        ):
            self.assertIn(arc_requirement, tutorial)


if __name__ == "__main__":
    unittest.main()
