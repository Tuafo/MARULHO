"""Dedicated tests for the Runtime Records module (living_loop_records).

Covers from_payload/to_payload round-trips for every dataclass,
enum value membership, and edge cases (empty payloads, missing keys,
invalid enum values).
"""
from __future__ import annotations

import unittest

from hecsn.cortex.episodic_memory import Provenance
from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    ActionVerificationRecord,
    ConsolidationRecord,
    ConsolidationStatus,
    PredictionRecord,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class PredictionStatusTests(unittest.TestCase):
    def test_all_members_present(self) -> None:
        expected = {"pending", "fulfilled", "contradicted", "unknown"}
        actual = {member.value for member in PredictionStatus}
        self.assertEqual(actual, expected)

    def test_string_enum_identity(self) -> None:
        self.assertEqual(PredictionStatus.PENDING, "pending")
        self.assertIsInstance(PredictionStatus.PENDING, str)


class ActionExecutionStatusTests(unittest.TestCase):
    def test_all_members_present(self) -> None:
        expected = {"requested", "executed", "failed", "reused"}
        actual = {member.value for member in ActionExecutionStatus}
        self.assertEqual(actual, expected)

    def test_string_enum_identity(self) -> None:
        self.assertEqual(ActionExecutionStatus.EXECUTED, "executed")


class VerificationStatusTests(unittest.TestCase):
    def test_all_members_present(self) -> None:
        expected = {"unknown", "unverified", "verified", "contradicted"}
        actual = {member.value for member in VerificationStatus}
        self.assertEqual(actual, expected)

    def test_string_enum_identity(self) -> None:
        self.assertEqual(VerificationStatus.VERIFIED, "verified")


class ConsolidationStatusTests(unittest.TestCase):
    def test_all_members_present(self) -> None:
        expected = {"raw", "cooling", "credited", "penalized", "forgiven", "consolidated", "retired"}
        actual = {member.value for member in ConsolidationStatus}
        self.assertEqual(actual, expected)

    def test_string_enum_identity(self) -> None:
        self.assertEqual(ConsolidationStatus.RAW, "raw")


# ---------------------------------------------------------------------------
# PredictionRecord round-trip tests
# ---------------------------------------------------------------------------
class PredictionRecordTests(unittest.TestCase):
    def test_round_trip_minimal_payload(self) -> None:
        record = PredictionRecord.from_payload({"predicted_outcome": "test"})
        round_tripped = PredictionRecord.from_payload(record.to_payload())
        self.assertEqual(record.prediction_id, round_tripped.prediction_id)
        self.assertEqual(record.predicted_outcome, "test")
        self.assertEqual(record.predicted_outcome, round_tripped.predicted_outcome)
        self.assertEqual(record.status, round_tripped.status)
        self.assertEqual(record.provenance, round_tripped.provenance)
        self.assertEqual(record.confidence, round_tripped.confidence)

    def test_round_trip_verified_prediction(self) -> None:
        payload = {
            "action_id": "act-1",
            "predicted_outcome": "Grounded evidence expected.",
            "verification": {
                "status": "verified",
                "success": True,
                "confidence": 0.9,
                "contradiction": False,
            },
        }
        record = PredictionRecord.from_payload(payload)
        self.assertEqual(record.status, PredictionStatus.FULFILLED)
        self.assertEqual(record.provenance, Provenance.VERIFIED)
        round_tripped = PredictionRecord.from_payload(record.to_payload())
        self.assertEqual(record.to_payload(), round_tripped.to_payload())

    def test_round_trip_contradicted_prediction(self) -> None:
        payload = {
            "action_id": "act-2",
            "predicted_outcome": "Expected missing file.",
            "verification": {
                "status": "contradicted",
                "success": False,
                "confidence": 0.8,
                "contradiction": True,
            },
        }
        record = PredictionRecord.from_payload(payload)
        self.assertEqual(record.status, PredictionStatus.CONTRADICTED)
        self.assertEqual(record.provenance, Provenance.CONTRADICTED)
        round_tripped = PredictionRecord.from_payload(record.to_payload())
        self.assertEqual(record.to_payload(), round_tripped.to_payload())

    def test_empty_payload_yields_defaults(self) -> None:
        record = PredictionRecord.from_payload({})
        self.assertNotEqual(record.prediction_id, "")
        self.assertEqual(record.predicted_outcome, "")
        self.assertEqual(record.status, PredictionStatus.UNKNOWN)
        self.assertEqual(record.confidence, 0.0)
        self.assertEqual(record.source_kind, "action")

    def test_explicit_status_without_verification(self) -> None:
        record = PredictionRecord.from_payload({"status": "fulfilled", "provenance": "verified"})
        self.assertEqual(record.status, PredictionStatus.FULFILLED)
        self.assertEqual(record.provenance, Provenance.VERIFIED)

    def test_topics_are_lowered(self) -> None:
        record = PredictionRecord.from_payload({"topics": ["Cats", "  MiCe  "]})
        self.assertIn("cats", record.topics)
        self.assertIn("mice", record.topics)

    def test_invalid_status_falls_back(self) -> None:
        record = PredictionRecord.from_payload({"status": "nonsense", "provenance": "inferred"})
        self.assertEqual(record.status, PredictionStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# ActionVerificationRecord round-trip tests
# ---------------------------------------------------------------------------
class ActionVerificationRecordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {
            "status": "verified",
            "success": True,
            "confidence": 0.95,
            "contradiction": False,
            "summary": "All checks passed.",
            "evidence": [{"path": "notes.md", "line": 5}],
        }
        record = ActionVerificationRecord.from_payload(payload, action_id="act-1")
        self.assertEqual(record.status, VerificationStatus.VERIFIED)
        self.assertTrue(record.success)
        round_tripped = ActionVerificationRecord.from_payload(record.to_payload(), action_id="act-1")
        self.assertEqual(record.verification_id, round_tripped.verification_id)
        self.assertEqual(record.to_payload()["status"], round_tripped.to_payload()["status"])
        self.assertEqual(record.to_payload()["evidence_count"], round_tripped.to_payload()["evidence_count"])

    def test_empty_payload_yields_defaults(self) -> None:
        record = ActionVerificationRecord.from_payload({})
        self.assertNotEqual(record.verification_id, "")
        self.assertEqual(record.status, VerificationStatus.UNKNOWN)
        self.assertFalse(record.success)
        self.assertEqual(record.evidence, ())

    def test_contradicted_defaults(self) -> None:
        record = ActionVerificationRecord.from_payload({"status": "contradicted"})
        self.assertFalse(record.success)
        self.assertTrue(record.contradiction)


# ---------------------------------------------------------------------------
# ActionExecutionRecord round-trip tests
# ---------------------------------------------------------------------------
class ActionExecutionRecordTests(unittest.TestCase):
    def test_round_trip_verified(self) -> None:
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
        self.assertEqual(record.action_id, round_tripped.action_id)
        self.assertEqual(record.prediction.status, PredictionStatus.FULFILLED)
        self.assertEqual(record.verification.status, VerificationStatus.VERIFIED)
        self.assertEqual(round_tripped.prediction.prediction_id, record.prediction.prediction_id)

    def test_round_trip_contradicted(self) -> None:
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

    def test_empty_payload_yields_requested_status(self) -> None:
        record = ActionExecutionRecord.from_payload({})
        self.assertEqual(record.execution_status, ActionExecutionStatus.REQUESTED)
        self.assertEqual(record.action_type, "")

    def test_explicit_execution_status(self) -> None:
        record = ActionExecutionRecord.from_payload({"execution_status": "failed"})
        self.assertEqual(record.execution_status, ActionExecutionStatus.FAILED)

    def test_has_outcome_implies_executed(self) -> None:
        record = ActionExecutionRecord.from_payload({"actual_outcome": "something happened"})
        self.assertEqual(record.execution_status, ActionExecutionStatus.EXECUTED)


# ---------------------------------------------------------------------------
# RuntimeEpisodeTrace round-trip tests
# ---------------------------------------------------------------------------
class RuntimeEpisodeTraceTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {
            "episode_id": "ep-1",
            "operation": "feed",
            "status": "succeeded",
            "created_at": "2026-01-01T00:00:00+00:00",
            "latency_ms": 100.0,
            "actual_output": {"tokens_processed": 50},
            "verification": {"status": "verified", "success": True},
        }
        record = RuntimeEpisodeTrace.from_payload(payload)
        round_tripped = RuntimeEpisodeTrace.from_payload(record.to_payload())
        self.assertEqual(record.episode_id, round_tripped.episode_id)
        self.assertEqual(record.operation, round_tripped.operation)
        self.assertEqual(record.latency_ms, round_tripped.latency_ms)

    def test_empty_payload_defaults(self) -> None:
        record = RuntimeEpisodeTrace.from_payload({})
        self.assertNotEqual(record.episode_id, "")
        self.assertEqual(record.operation, "unknown")
        self.assertEqual(record.status, "succeeded")  # no failure → succeeded

    def test_failure_implies_failed_status(self) -> None:
        record = RuntimeEpisodeTrace.from_payload({"failure": {"reason": "timeout"}})
        self.assertEqual(record.status, "failed")

    def test_prediction_record_returns_none_when_no_predicted_text(self) -> None:
        record = RuntimeEpisodeTrace.from_payload({"episode_id": "ep-1", "operation": "feed"})
        self.assertIsNone(record.prediction_record())

    def test_prediction_record_returns_record_when_predicted_text_present(self) -> None:
        record = RuntimeEpisodeTrace.from_payload({
            "episode_id": "ep-1",
            "operation": "query",
            "prediction": {"predicted_output": "Some answer"},
            "verification": {"status": "verified", "success": True},
        })
        pred = record.prediction_record()
        self.assertIsNotNone(pred)
        self.assertEqual(pred.predicted_outcome, "Some answer")  # type: ignore[union-attr]
        self.assertEqual(pred.status, PredictionStatus.FULFILLED)  # type: ignore[union-attr]

    def test_invalid_latency_becomes_none(self) -> None:
        record = RuntimeEpisodeTrace.from_payload({"latency_ms": "not_a_number"})
        self.assertIsNone(record.latency_ms)


# ---------------------------------------------------------------------------
# SkillMemoryRecord round-trip tests
# ---------------------------------------------------------------------------
class SkillMemoryRecordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {
            "skill_id": "skill-1",
            "action_type": "workspace_search",
            "tool": "workspace_search",
            "action_count": 5,
            "success_count": 3,
            "failure_count": 2,
            "success_rate": 0.6,
            "provenance": "verified",
            "topics": ["cats"],
        }
        record = SkillMemoryRecord.from_payload(payload)
        round_tripped = SkillMemoryRecord.from_payload(record.to_payload())
        self.assertEqual(record.skill_id, round_tripped.skill_id)
        self.assertEqual(record.action_count, round_tripped.action_count)
        self.assertAlmostEqual(record.success_rate, round_tripped.success_rate)

    def test_empty_payload_yields_defaults(self) -> None:
        record = SkillMemoryRecord.from_payload({})
        self.assertNotEqual(record.skill_id, "")
        self.assertEqual(record.action_count, 0)
        self.assertEqual(record.success_rate, 0.0)
        self.assertEqual(record.status, "unverified")

    def test_from_action_records_groups_by_type(self) -> None:
        actions = tuple(
            ActionExecutionRecord.from_payload(p)
            for p in [
                {
                    "action_id": "act-search-ok",
                    "action_type": "workspace_search",
                    "inputs": {"query_text": "cats"},
                    "predicted_outcome": "Find cat notes.",
                    "actual_outcome": "Found cat notes.",
                    "verification": {"status": "verified", "success": True, "confidence": 0.9},
                    "topics": ["cats"],
                    "recorded_at": "2026-01-02T00:00:00+00:00",
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
                },
            ]
        )
        memories = SkillMemoryRecord.from_action_records(actions)
        self.assertEqual(len(memories), 1)
        search = memories[0]
        self.assertEqual(search.action_type, "workspace_search")
        self.assertEqual(search.action_count, 2)
        self.assertEqual(search.success_count, 1)
        self.assertEqual(search.failure_count, 1)
        self.assertAlmostEqual(search.success_rate, 0.5)
        self.assertEqual(search.status, "mixed")

    def test_success_rate_clamped_to_01(self) -> None:
        record = SkillMemoryRecord.from_payload({"success_rate": 2.0})
        self.assertEqual(record.success_rate, 1.0)
        record = SkillMemoryRecord.from_payload({"success_rate": -1.0})
        self.assertEqual(record.success_rate, 0.0)


# ---------------------------------------------------------------------------
# ProvenanceState round-trip tests
# ---------------------------------------------------------------------------
class ProvenanceStateTests(unittest.TestCase):
    def test_round_trip_from_distribution(self) -> None:
        state = ProvenanceState.from_distribution(
            {"observed": 2, "inferred": 3, "dreamed": 4, "verified": 5, "contradicted": 6}
        )
        payload = state.to_payload()
        round_tripped = ProvenanceState.from_payload(payload)
        self.assertEqual(state.observed, round_tripped.observed)
        self.assertEqual(state.total, round_tripped.total)
        self.assertEqual(payload["observed"], 2)
        self.assertEqual(payload["total"], 20)

    def test_empty_distribution_yields_zeros(self) -> None:
        state = ProvenanceState.from_distribution({})
        self.assertEqual(state.observed, 0)
        self.assertEqual(state.total, 0)

    def test_none_payload_yields_zeros(self) -> None:
        state = ProvenanceState.from_payload(None)
        self.assertEqual(state.total, 0)

    def test_negative_values_clamped_to_zero(self) -> None:
        state = ProvenanceState.from_distribution({"observed": -5, "verified": 3})
        self.assertEqual(state.observed, 0)
        self.assertEqual(state.verified, 3)


# ---------------------------------------------------------------------------
# ConsolidationRecord round-trip tests
# ---------------------------------------------------------------------------
class ConsolidationRecordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {
            "consolidation_id": "con-1",
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
        record = ConsolidationRecord.from_payload(payload)
        round_tripped = ConsolidationRecord.from_payload(record.to_payload())
        self.assertEqual(record.consolidation_id, round_tripped.consolidation_id)
        self.assertEqual(record.status, round_tripped.status)
        # Note: semantic_terms is populated from query_terms in from_payload,
        # but to_payload outputs it as semantic_terms. A full round-trip
        # requires re-adding query_terms to the payload.
        self.assertEqual(record.consolidation_states, round_tripped.consolidation_states)

    def test_round_trip_with_query_terms_key(self) -> None:
        """Round-trip works when to_payload output is fed back with query_terms."""
        payload = {
            "consolidation_id": "con-1",
            "query_text": "cats chase mice",
            "query_terms": ["cats", "mice"],
            "source_weights": {"notes.md": 1.0},
            "provider_weights": {"workspace": 1.0},
            "credit_events": 1,
            "penalty_events": 1,
            "aggregate_count": 2,
        }
        record = ConsolidationRecord.from_payload(payload)
        # Re-add query_terms to the to_payload output for round-trip
        round_payload = record.to_payload()
        round_payload["query_terms"] = round_payload.pop("semantic_terms")
        round_tripped = ConsolidationRecord.from_payload(round_payload)
        self.assertEqual(record.semantic_terms, round_tripped.semantic_terms)

    def test_empty_payload_yields_raw_status(self) -> None:
        record = ConsolidationRecord.from_payload({})
        self.assertNotEqual(record.consolidation_id, "")
        self.assertEqual(record.status, ConsolidationStatus.RAW)
        self.assertEqual(record.aggregate_count, 1)

    def test_credit_events_imply_credited_status(self) -> None:
        record = ConsolidationRecord.from_payload({"credit_events": 3})
        self.assertEqual(record.status, ConsolidationStatus.CREDITED)

    def test_penalized_when_penalties_exceed_credits(self) -> None:
        record = ConsolidationRecord.from_payload({"credit_events": 1, "penalty_events": 5})
        self.assertEqual(record.status, ConsolidationStatus.PENALIZED)

    def test_forgiven_status(self) -> None:
        record = ConsolidationRecord.from_payload({"forgiveness_events": 2, "credit_events": 0, "penalty_events": 0})
        self.assertEqual(record.status, ConsolidationStatus.FORGIVEN)

    def test_consolidated_status_when_aggregate_gt_1(self) -> None:
        record = ConsolidationRecord.from_payload({"aggregate_count": 3})
        self.assertEqual(record.status, ConsolidationStatus.CONSOLIDATED)

    def test_explicit_status_overrides_inferred(self) -> None:
        record = ConsolidationRecord.from_payload({"status": "retired", "credit_events": 5})
        self.assertEqual(record.status, ConsolidationStatus.RETIRED)

    def test_semantic_state_flags(self) -> None:
        record = ConsolidationRecord.from_payload(
            {
                "query_text": "cats",
                "source_weights": {"notes.md": 1.0},
                "credit_events": 1,
                "penalty_events": 1,
                "aggregate_count": 2,
            }
        )
        payload = record.to_payload()
        self.assertIn("observed", payload["consolidation_states"])
        self.assertIn("summarized", payload["consolidation_states"])
        self.assertIn("verified", payload["consolidation_states"])
        self.assertIn("contradicted", payload["consolidation_states"])
        self.assertIn("semanticized", payload["consolidation_states"])

    def test_record_id_alias_for_consolidation_id(self) -> None:
        record = ConsolidationRecord.from_payload({"record_id": "delayed-1", "query_text": "x"})
        self.assertEqual(record.consolidation_id, "delayed-1")


# ---------------------------------------------------------------------------
# Re-export shim verification: all symbols must remain importable from
# hecsn.service.living_loop
# ---------------------------------------------------------------------------
class ReExportShimTests(unittest.TestCase):
    def test_records_importable_from_living_loop(self) -> None:
        from hecsn.service.living_loop import (
            ActionExecutionRecord as LL_AER,
            ActionExecutionStatus as LL_AES,
            ActionVerificationRecord as LL_AVR,
            ConsolidationRecord as LL_CR,
            ConsolidationStatus as LL_CS,
            PredictionRecord as LL_PR,
            PredictionStatus as LL_PS,
            ProvenanceState as LL_ProS,
            RuntimeEpisodeTrace as LL_RET,
            SkillMemoryRecord as LL_SMR,
            VerificationStatus as LL_VS,
        )

        # Verify they are the exact same objects as in the records module
        self.assertIs(LL_AER, ActionExecutionRecord)
        self.assertIs(LL_AES, ActionExecutionStatus)
        self.assertIs(LL_AVR, ActionVerificationRecord)
        self.assertIs(LL_CR, ConsolidationRecord)
        self.assertIs(LL_CS, ConsolidationStatus)
        self.assertIs(LL_PR, PredictionRecord)
        self.assertIs(LL_PS, PredictionStatus)
        self.assertIs(LL_ProS, ProvenanceState)
        self.assertIs(LL_RET, RuntimeEpisodeTrace)
        self.assertIs(LL_SMR, SkillMemoryRecord)
        self.assertIs(LL_VS, VerificationStatus)
