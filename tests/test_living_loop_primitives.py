from __future__ import annotations

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
    SkillMemoryRecord,
    VerificationStatus,
)
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
        self.assertEqual(set(endpoint_loop["provenance"]["distribution"].keys()), {
            "observed",
            "inferred",
            "dreamed",
            "verified",
            "contradicted",
        })

    def test_tutorial_documents_living_loop_endpoint_and_arc_benchmark_boundary(self) -> None:
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
        ):
            self.assertIn(field, tutorial)
        for arc_requirement in (
            "ARC-AGI should remain a **separate benchmark path**",
            "object parser",
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
