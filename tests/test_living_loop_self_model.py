"""Dedicated tests for the Operational Self-Model module (living_loop_self_model.py).

Covers:
- OperationalSelfModel.build with known inputs
- OperationalSelfModel.from_payload round-trip
- OperationalSelfModel.to_payload structure and surface methods
- build_runtime_benchmark_telemetry with synthetic runtime samples
- Telemetry helpers: _endpoint_bucket_name, _latency_summary,
  _extract_cache_summary, _extract_nim_summary, _memory_counter_summary,
  _endpoint_latency_empty
- Re-export verification: symbols still importable from living_loop
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    ConsolidationRecord,
    PredictionRecord,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
)
from hecsn.service.living_loop_policy import (
    WorldModelLiteSummary,
)

# ---------------------------------------------------------------------------
# Import from the new Self-Model module
# ---------------------------------------------------------------------------
from hecsn.service.living_loop_self_model import (
    OperationalSelfModel,
    build_runtime_benchmark_telemetry,
    _endpoint_bucket_name,
    _endpoint_latency_empty,
    _extract_cache_summary,
    _extract_nim_summary,
    _latency_summary,
    _memory_counter_summary,
)


# ---------------------------------------------------------------------------
# Telemetry helper tests
# ---------------------------------------------------------------------------

class TestEndpointLatencyEmpty(unittest.TestCase):
    """_endpoint_latency_empty returns the expected zero-state structure."""

    def test_structure(self) -> None:
        result = _endpoint_latency_empty()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["latency_count"], 0)
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["failure_count"], 0)
        self.assertIsNone(result["min_ms"])
        self.assertIsNone(result["avg_ms"])
        self.assertIsNone(result["max_ms"])
        self.assertEqual(result["success_rate"], 0.0)


class TestEndpointBucketName(unittest.TestCase):
    """_endpoint_bucket_name normalizes operation names into buckets."""

    def test_feed(self) -> None:
        self.assertEqual(_endpoint_bucket_name("feed"), "feed")

    def test_query(self) -> None:
        self.assertEqual(_endpoint_bucket_name("query"), "query")

    def test_respond(self) -> None:
        self.assertEqual(_endpoint_bucket_name("respond"), "respond")

    def test_runtime_action_aliases(self) -> None:
        for name in ("action", "runtime_action", "digital_action", "terminus_action"):
            self.assertEqual(_endpoint_bucket_name(name), "runtime_action")

    def test_unknown_operation(self) -> None:
        self.assertEqual(_endpoint_bucket_name("custom_op"), "custom_op")

    def test_empty_string_returns_unknown(self) -> None:
        self.assertEqual(_endpoint_bucket_name(""), "unknown")

    def test_none_returns_unknown(self) -> None:
        self.assertEqual(_endpoint_bucket_name(None), "unknown")


class TestLatencySummary(unittest.TestCase):
    """_latency_summary computes latency statistics from raw values."""

    def test_basic_summary(self) -> None:
        result = _latency_summary(3, 2, 1, [100.0, 200.0, 300.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["latency_count"], 3)
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["min_ms"], 100.0)
        self.assertAlmostEqual(result["avg_ms"], 200.0)
        self.assertEqual(result["max_ms"], 300.0)
        self.assertAlmostEqual(result["success_rate"], 2.0 / 3.0)

    def test_empty_latencies(self) -> None:
        result = _latency_summary(0, 0, 0, [])
        self.assertEqual(result["latency_count"], 0)
        self.assertIsNone(result["min_ms"])
        self.assertIsNone(result["avg_ms"])
        self.assertIsNone(result["max_ms"])
        self.assertEqual(result["success_rate"], 0.0)

    def test_negative_latencies_excluded(self) -> None:
        result = _latency_summary(2, 1, 1, [100.0, -50.0])
        self.assertEqual(result["latency_count"], 1)
        self.assertEqual(result["min_ms"], 100.0)
        self.assertEqual(result["max_ms"], 100.0)

    def test_zero_counts_clamped(self) -> None:
        result = _latency_summary(-1, -1, -1, [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["failure_count"], 0)


class TestExtractCacheSummary(unittest.TestCase):
    """_extract_cache_summary extracts cache hit/miss statistics."""

    def test_tracked_cache(self) -> None:
        result = _extract_cache_summary({"cache_hits": 3, "cache_misses": 1, "cache_size": 5})
        self.assertTrue(result["tracked"])
        self.assertAlmostEqual(result["hit_rate"], 0.75)
        self.assertEqual(result["hit_count"], 3)
        self.assertEqual(result["miss_count"], 1)
        self.assertEqual(result["cache_size"], 5)

    def test_untracked_cache(self) -> None:
        result = _extract_cache_summary({})
        self.assertFalse(result["tracked"])
        self.assertIsNone(result["hit_rate"])
        self.assertIsNone(result["hit_count"])
        self.assertIsNone(result["miss_count"])

    def test_empty_stats_default_size(self) -> None:
        result = _extract_cache_summary({"cache_size": 0})
        self.assertEqual(result["cache_size"], 0)


class TestExtractNimSummary(unittest.TestCase):
    """_extract_nim_summary extracts NIM call statistics from cortex data."""

    def test_full_cortex(self) -> None:
        cortex = {
            "enabled": True,
            "thoughts_generated": 2,
            "dreams_generated": 1,
            "episodic_memory": {
                "embedder": {
                    "kind": "NIMEmbedder",
                    "nim_calls": 4,
                    "rate_limit_hits": 1,
                    "available": True,
                    "degraded": False,
                    "fallback_calls": 0,
                    "error_calls": 2,
                },
            },
        }
        result = _extract_nim_summary(cortex)
        self.assertTrue(result["available"])
        self.assertEqual(result["chat_generations_observed"], 3)
        self.assertEqual(result["embedding_nim_calls"], 4)
        self.assertEqual(result["observed_call_count"], 7)
        self.assertIsNone(result["calls_per_minute"])
        self.assertEqual(result["rate_limit_hits"], 1)
        embedder = result["embedder"]
        self.assertEqual(embedder["kind"], "NIMEmbedder")
        self.assertTrue(embedder["available"])
        self.assertFalse(embedder["degraded"])
        self.assertEqual(embedder["error_calls"], 2)

    def test_none_cortex(self) -> None:
        result = _extract_nim_summary(None)
        self.assertFalse(result["available"])
        self.assertEqual(result["observed_call_count"], 0)

    def test_empty_cortex(self) -> None:
        result = _extract_nim_summary({})
        self.assertFalse(result["available"])
        self.assertEqual(result["chat_generations_observed"], 0)


class TestMemoryCounterSummary(unittest.TestCase):
    """_memory_counter_summary extracts memory fill and capacity information."""

    def test_available_memory(self) -> None:
        result = _memory_counter_summary(
            {"size": 4, "capacity": 16, "fill_ratio": 0.25},
            None,
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["size"], 4)
        self.assertEqual(result["capacity"], 16)
        self.assertAlmostEqual(result["fill_ratio"], 0.25)
        self.assertEqual(result["source"], "cortex_episodic_memory")

    def test_capacity_pressure(self) -> None:
        result = _memory_counter_summary(
            {"size": 15, "capacity": 16, "fill_ratio": 0.95},
            None,
        )
        self.assertEqual(result["status"], "capacity_pressure")

    def test_runtime_memory_takes_priority(self) -> None:
        result = _memory_counter_summary(
            {"size": 1, "capacity": 2, "fill_ratio": 0.5},
            {"size": 10, "capacity": 20, "fill_ratio": 0.5},
        )
        self.assertEqual(result["size"], 10)
        self.assertEqual(result["source"], "runtime_memory_store")

    def test_no_memory_snapshot(self) -> None:
        result = _memory_counter_summary(None, None)
        self.assertEqual(result["status"], "no_memory_snapshot")
        self.assertEqual(result["source"], "unavailable")

    def test_no_fill_ratio(self) -> None:
        result = _memory_counter_summary({"size": 4}, None)
        self.assertEqual(result["status"], "no_capacity_snapshot")


# ---------------------------------------------------------------------------
# build_runtime_benchmark_telemetry tests
# ---------------------------------------------------------------------------

class TestBuildRuntimeBenchmarkTelemetry(unittest.TestCase):
    """build_runtime_benchmark_telemetry produces telemetry from runtime facts."""

    def _make_episodes(self) -> tuple[RuntimeEpisodeTrace, ...]:
        return tuple(
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
                    "verification": {
                        "status": "contradicted",
                        "success": False,
                        "contradiction": True,
                    },
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

    def _make_actions(self) -> tuple[ActionExecutionRecord, ...]:
        return tuple(
            ActionExecutionRecord.from_payload(payload)
            for payload in [
                {
                    "action_id": "act-ok",
                    "action_type": "workspace_search",
                    "predicted_outcome": "Find evidence.",
                    "verification": {
                        "status": "verified",
                        "success": True,
                        "confidence": 0.8,
                    },
                },
                {
                    "action_id": "act-bad",
                    "action_type": "workspace_read",
                    "predicted_outcome": "Read missing evidence.",
                    "execution_status": "failed",
                    "verification": {
                        "status": "contradicted",
                        "success": False,
                        "contradiction": True,
                    },
                },
            ]
        )

    def test_endpoint_latency_counts(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
        )
        self.assertEqual(telemetry["endpoint_latency_ms"]["feed"]["count"], 1)
        self.assertEqual(
            telemetry["endpoint_latency_ms"]["query"]["avg_ms"], 200.0
        )
        self.assertEqual(
            telemetry["endpoint_latency_ms"]["respond"]["failure_count"], 1
        )
        self.assertEqual(
            telemetry["endpoint_latency_ms"]["runtime_action"]["latency_count"], 1
        )
        self.assertEqual(
            telemetry["endpoint_latency_ms"]["runtime_action"]["count"], 3
        )

    def test_tokens_per_second_from_episode_traces(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
        )
        self.assertAlmostEqual(telemetry["tokens_per_second"]["value"], 500.0)
        self.assertEqual(
            telemetry["tokens_per_second"]["source"], "runtime_episode_traces"
        )

    def test_memory_nim_and_cache(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
            action_loop={"actions_recorded": 2},
            memory={
                "size": 4,
                "capacity": 16,
                "fill_ratio": 0.25,
                "total_stored": 4,
                "total_evicted": 0,
            },
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
                    },
                },
            },
        )
        self.assertEqual(telemetry["memory"]["status"], "available")
        self.assertEqual(telemetry["nim"]["observed_call_count"], 7)
        self.assertIsNone(telemetry["nim"]["calls_per_minute"])
        self.assertEqual(telemetry["nim"]["rate_limit_hits"], 1)
        self.assertAlmostEqual(telemetry["cache"]["hit_rate"], 0.75)

    def test_action_and_verification_success_rates(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
        )
        self.assertAlmostEqual(telemetry["action_success"]["success_rate"], 0.5)
        self.assertAlmostEqual(
            telemetry["verification_success"]["success_rate"], 0.5
        )

    def test_policy_recommendations(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
        )
        self.assertEqual(
            telemetry["policy_recommendations"]["counts"]["investigate_contradictions"],
            1,
        )

    def test_replay_sample_summary(self) -> None:
        episodes = self._make_episodes()
        actions = self._make_actions()
        world = WorldModelLiteSummary.from_records(actions=actions)
        telemetry = build_runtime_benchmark_telemetry(
            runtime_episodes=episodes,
            actions=actions,
            world_model_lite=world,
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
                    "safety_flags": {
                        "audit_only": True,
                        "external_calls_made": False,
                    },
                },
                "safety_flags": {
                    "audit_only": True,
                    "external_calls_made": False,
                },
            },
        )
        self.assertEqual(telemetry["replay_sample_summary"]["count"], 2)
        self.assertEqual(
            telemetry["replay_sample_summary"]["mode_counts"]["execute"], 1
        )
        self.assertEqual(
            telemetry["replay_sample_summary"]["status_counts"]["recorded"], 2
        )
        self.assertEqual(
            telemetry["replay_sample_summary"]["latest_history_item"][
                "replay_sample_id"
            ],
            "replay-execute-1",
        )
        self.assertTrue(
            telemetry["replay_sample_summary"]["safety_flags"]["audit_only"]
        )
        self.assertFalse(
            telemetry["replay_sample_summary"]["safety_flags"]["external_calls_made"]
        )

    def test_empty_inputs_produce_valid_structure(self) -> None:
        telemetry = build_runtime_benchmark_telemetry()
        self.assertEqual(telemetry["schema_version"], 1)
        self.assertIn("generated_at", telemetry)
        self.assertEqual(telemetry["sample"]["runtime_episode_count"], 0)
        self.assertEqual(telemetry["sample"]["action_count"], 0)
        self.assertEqual(telemetry["action_success"]["success_rate"], 0.0)
        self.assertIsNone(telemetry["tokens_per_second"]["value"])


# ---------------------------------------------------------------------------
# OperationalSelfModel tests
# ---------------------------------------------------------------------------

class TestOperationalSelfModelBuild(unittest.TestCase):
    """OperationalSelfModel.build produces a valid model from explicit inputs."""

    def _make_actions(self) -> tuple[ActionExecutionRecord, ...]:
        return tuple(
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

    def test_build_produces_model_with_all_fields(self) -> None:
        actions = self._make_actions()
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
                "supported_actions": [
                    "workspace_search",
                    "workspace_read",
                    "web_fetch",
                ],
            },
            memory={
                "size": 4,
                "capacity": 16,
                "fill_ratio": 0.25,
                "total_stored": 4,
                "total_evicted": 0,
                "provenance_distribution": {
                    "verified": 1,
                    "contradicted": 1,
                },
            },
            cortex={"enabled": True, "memory_count": 4},
            generated_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(model.token_count, 12)
        self.assertEqual(model.state_revision, 3)
        self.assertTrue(model.configured)
        self.assertTrue(model.running)
        self.assertEqual(len(model.actions), 2)
        self.assertIsNotNone(model.world_model_lite)

    def test_build_with_minimal_inputs(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=False,
            running=False,
            provenance=ProvenanceState(),
        )
        self.assertFalse(model.configured)
        self.assertFalse(model.running)
        self.assertEqual(len(model.predictions), 0)
        self.assertEqual(len(model.actions), 0)


class TestOperationalSelfModelFromPayload(unittest.TestCase):
    """OperationalSelfModel.from_payload reconstructs a model from a payload."""

    def test_round_trip_via_to_payload(self) -> None:
        actions = tuple(
            ActionExecutionRecord.from_payload(payload)
            for payload in [
                {
                    "action_id": "act-verified",
                    "action_type": "workspace_search",
                    "predicted_outcome": "Find cat evidence.",
                    "actual_outcome": "Found cat evidence.",
                    "verification": {
                        "status": "verified",
                        "success": True,
                    },
                    "topics": ["cats"],
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                },
            ]
        )
        original = OperationalSelfModel.build(
            token_count=5,
            state_revision=1,
            configured=True,
            running=True,
            provenance=ProvenanceState(verified=1, total=1),
            actions=actions,
            memory={"size": 2, "capacity": 8, "fill_ratio": 0.25},
            generated_at="2026-01-01T00:00:00+00:00",
        )
        payload = original.to_payload()
        restored = OperationalSelfModel.from_payload(payload)

        self.assertEqual(restored.token_count, original.token_count)
        self.assertEqual(restored.state_revision, original.state_revision)
        self.assertEqual(restored.configured, original.configured)
        self.assertEqual(restored.running, original.running)
        self.assertEqual(len(restored.actions), len(original.actions))
        self.assertEqual(
            restored.provenance.verified, original.provenance.verified
        )
        self.assertEqual(restored.generated_at, original.generated_at)

    def test_from_payload_with_empty_payload(self) -> None:
        model = OperationalSelfModel.from_payload({})
        self.assertEqual(model.token_count, 0)
        self.assertEqual(model.state_revision, 0)
        self.assertFalse(model.configured)
        self.assertFalse(model.running)

    def test_from_payload_with_none_payload(self) -> None:
        model = OperationalSelfModel.from_payload(None)
        self.assertEqual(model.token_count, 0)

    def test_from_payload_merges_episode_predictions(self) -> None:
        """Episode predictions not already in explicit predictions are merged."""
        payload = {
            "predictions": [
                {
                    "prediction_id": "pred-explicit",
                    "source_id": "source-1",
                    "predicted_outcome": "Explicit prediction.",
                    "status": "pending",
                },
            ],
            "runtime_episodes": [
                {
                    "episode_id": "ep-1",
                    "operation": "feed",
                    "status": "succeeded",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "prediction": {
                        "predicted_output": "Episode prediction.",
                    },
                    "verification": {"status": "verified", "success": True},
                },
            ],
            "provenance": {},
        }
        model = OperationalSelfModel.from_payload(payload)
        pred_ids = {p.prediction_id for p in model.predictions}
        self.assertIn("pred-explicit", pred_ids)
        # Episode-derived prediction has a stable ID starting with "pred-"
        episode_pred_ids = pred_ids - {"pred-explicit"}
        self.assertEqual(len(episode_pred_ids), 1)
        episode_pred_id = episode_pred_ids.pop()
        self.assertTrue(episode_pred_id.startswith("pred-"))


class TestOperationalSelfModelToPayload(unittest.TestCase):
    """OperationalSelfModel.to_payload produces a complete payload dict."""

    def _make_model(self) -> OperationalSelfModel:
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
        return OperationalSelfModel.build(
            token_count=12,
            state_revision=3,
            configured=True,
            running=True,
            provenance=ProvenanceState(verified=1, contradicted=1, total=2),
            predictions=[item.prediction for item in actions],
            actions=actions,
            action_loop={
                "actions_recorded": 10,
                "supported_actions": [
                    "workspace_search",
                    "workspace_read",
                    "web_fetch",
                ],
            },
            memory={
                "size": 4,
                "capacity": 16,
                "fill_ratio": 0.25,
                "total_stored": 4,
                "total_evicted": 0,
                "provenance_distribution": {
                    "verified": 1,
                    "contradicted": 1,
                },
            },
            cortex={"enabled": True, "memory_count": 4},
            generated_at="2026-01-01T00:00:00+00:00",
        )

    def test_payload_includes_capabilities(self) -> None:
        payload = self._make_model().to_payload()
        self.assertIn("digital_action_execution", payload["capabilities"])
        self.assertIn("runtime_configured", payload["capabilities"])
        self.assertIn("runtime_running", payload["capabilities"])
        self.assertIn("prediction_tracking", payload["capabilities"])
        self.assertIn("world_model_lite_policy_scoring", payload["capabilities"])

    def test_payload_limits(self) -> None:
        payload = self._make_model().to_payload()
        self.assertEqual(payload["limits"]["action_history_count"], 10)
        self.assertTrue(payload["limits"]["action_history_truncated"])
        self.assertIn("workspace_search", payload["limits"]["supported_actions"])

    def test_payload_budgets(self) -> None:
        payload = self._make_model().to_payload()
        self.assertEqual(payload["budgets"]["memory_size"], 4)
        self.assertEqual(payload["budgets"]["action_history_used"], 10)

    def test_payload_memory_health(self) -> None:
        payload = self._make_model().to_payload()
        self.assertEqual(payload["memory_health"]["status"], "available")
        self.assertEqual(payload["memory_health"]["capacity"], 16)
        self.assertAlmostEqual(payload["memory_health"]["fill_ratio"], 0.25)

    def test_payload_grounding_health_contradictions(self) -> None:
        payload = self._make_model().to_payload()
        self.assertEqual(
            payload["grounding_health"]["status"], "contradictions_present"
        )
        self.assertEqual(
            payload["grounding_health"]["contradicted_action_count"], 1
        )

    def test_payload_recent_failures(self) -> None:
        payload = self._make_model().to_payload()
        self.assertTrue(len(payload["recent_failures"]) >= 1)
        self.assertEqual(
            payload["recent_failures"][0]["action_id"], "act-contradicted"
        )
        self.assertEqual(
            payload["recent_failures"][0]["verification_status"], "contradicted"
        )

    def test_payload_uncertain_domains(self) -> None:
        payload = self._make_model().to_payload()
        self.assertTrue(len(payload["uncertain_domains"]) >= 1)
        self.assertEqual(payload["uncertain_domains"][0]["domain"], "missing")

    def test_payload_tools(self) -> None:
        payload = self._make_model().to_payload()
        tool_names = {item["name"] for item in payload["tools"]}
        self.assertEqual(
            tool_names, {"workspace_search", "workspace_read", "web_fetch"}
        )

    def test_payload_skill_memories(self) -> None:
        payload = self._make_model().to_payload()
        skill_by_tool = {item["tool"]: item for item in payload["skill_memories"]}
        self.assertEqual(skill_by_tool["workspace_search"]["status"], "verified")
        self.assertEqual(
            skill_by_tool["workspace_read"]["status"], "contradicted"
        )

    def test_payload_benchmark_telemetry(self) -> None:
        payload = self._make_model().to_payload()
        self.assertIn("benchmark_telemetry", payload)
        bt = payload["benchmark_telemetry"]
        self.assertEqual(bt["schema_version"], 1)
        self.assertIn("endpoint_latency_ms", bt)
        self.assertIn("tokens_per_second", bt)

    def test_payload_top_level_fields(self) -> None:
        payload = self._make_model().to_payload()
        self.assertEqual(payload["token_count"], 12)
        self.assertEqual(payload["state_revision"], 3)
        self.assertTrue(payload["configured"])
        self.assertTrue(payload["running"])
        self.assertEqual(payload["prediction_count"], 2)
        self.assertEqual(payload["action_count"], 2)
        self.assertEqual(payload["consolidation_count"], 0)

    def test_payload_world_model_lite(self) -> None:
        payload = self._make_model().to_payload()
        self.assertIn("world_model_lite", payload)
        wml = payload["world_model_lite"]
        self.assertIsInstance(wml, dict)


class TestOperationalSelfModelSurfaceMethods(unittest.TestCase):
    """Surface methods on OperationalSelfModel produce correct derived data."""

    def test_surface_uncertain_domains_pending_prediction(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=False,
            running=False,
            provenance=ProvenanceState(),
            predictions=[
                PredictionRecord(
                    prediction_id="pred-1",
                    source_id="src-1",
                    predicted_outcome="Test.",
                    status=PredictionStatus.PENDING,
                    topics=("physics",),
                ),
            ],
        )
        domains = model._surface_uncertain_domains()
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0]["domain"], "physics")
        self.assertEqual(domains[0]["pending_predictions"], 1)

    def test_surface_memory_health_pressure(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=False,
            running=False,
            provenance=ProvenanceState(),
            memory={"fill_ratio": 0.95, "capacity": 100, "size": 95},
        )
        health = model._surface_memory_health()
        self.assertEqual(health["status"], "capacity_pressure")
        self.assertAlmostEqual(health["fill_ratio"], 0.95)

    def test_surface_grounding_health_no_grounding(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=False,
            running=False,
            provenance=ProvenanceState(),
        )
        health = model._surface_grounding_health(
            model.world_model_lite or WorldModelLiteSummary()
        )
        self.assertEqual(health["status"], "no_grounding_observed")

    def test_surface_capabilities_includes_cortex(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=True,
            running=True,
            provenance=ProvenanceState(),
            cortex={"enabled": True},
        )
        skills = model._surface_skill_memories()
        caps = model._surface_capabilities(
            skill_memories=skills,
            world_model_lite=model.world_model_lite or WorldModelLiteSummary(),
        )
        self.assertIn("cortex_loop_snapshot", caps)

    def test_surface_limits_no_truncation(self) -> None:
        model = OperationalSelfModel.build(
            token_count=0,
            state_revision=0,
            configured=False,
            running=False,
            provenance=ProvenanceState(),
            action_loop={"actions_recorded": 0},
        )
        skills = model._surface_skill_memories()
        limits = model._surface_limits(skills)
        self.assertFalse(limits["action_history_truncated"])


# ---------------------------------------------------------------------------
# Re-export verification
# ---------------------------------------------------------------------------

class TestSelfModelReExports(unittest.TestCase):
    """Verify that symbols are still importable from the living_loop shim."""

    def test_operational_self_model_importable_from_living_loop(self) -> None:
        from hecsn.service.living_loop import OperationalSelfModel as OSM

        self.assertIs(OSM, OperationalSelfModel)

    def test_build_runtime_benchmark_telemetry_importable(self) -> None:
        from hecsn.service.living_loop import (
            build_runtime_benchmark_telemetry as brbt,
        )

        self.assertIs(brbt, build_runtime_benchmark_telemetry)

    def test_telemetry_helpers_importable(self) -> None:
        from hecsn.service.living_loop import (
            _endpoint_bucket_name,
            _endpoint_latency_empty,
            _extract_cache_summary,
            _extract_nim_summary,
            _latency_summary,
            _memory_counter_summary,
        )

        # Verify they are callable
        self.assertTrue(callable(_endpoint_bucket_name))
        self.assertTrue(callable(_endpoint_latency_empty))
        self.assertTrue(callable(_extract_cache_summary))
        self.assertTrue(callable(_extract_nim_summary))
        self.assertTrue(callable(_latency_summary))
        self.assertTrue(callable(_memory_counter_summary))


if __name__ == "__main__":
    unittest.main()
