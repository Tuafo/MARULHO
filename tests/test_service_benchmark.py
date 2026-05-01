from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from fastapi import FastAPI

from hecsn.evaluation.service_benchmark import (
    benchmark_service_app,
    create_tiny_service_benchmark_checkpoint,
    run_service_benchmark,
)


def _scratch_root(name: str) -> Path:
    root = Path("reports") / "service_benchmark_tests" / f"{name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _fake_service_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def runtime_truth() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "verdict": "alive",
            "recommended_action": "continue_monitoring",
            "cortex_available": True,
            "memory_pressure": {"fill_fraction": 0.1, "size": 2, "capacity": 64, "pressure": "low"},
            "replay_role": "preview_only_not_training",
            "safety_flags": {"replay_dataset_preview_only": True, "training_started": False},
            "latency_ms": {"last_tick": 1.0, "tokens_per_second": 10.0},
            "evidence": {"configured": True, "running": True, "token_count": 12},
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        return {
            "status": "ok",
            "token_count": 12,
            "runtime_truth": runtime_truth(),
        }

    @app.get("/terminus")
    def terminus() -> dict[str, Any]:
        return {
            "terminus_runtime": {"configured": True, "running": True},
            "runtime_truth": runtime_truth(),
        }

    @app.post("/feed")
    def feed(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "feed_summary": {"tokens_processed": len(str(payload.get("text", "")).split())},
            "runtime_episode": {"operation": "feed"},
            "dirty_state": True,
            "state_revision": 1,
        }

    @app.post("/query")
    def query(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_summary": {"query_text": payload.get("query_text")},
            "concept_summary": {},
            "gap_plan": {},
            "service_state": {},
            "runtime_episode": {"operation": "query"},
        }

    @app.post("/respond")
    def respond(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "trace_id": "trace-1",
            "trace_path": "trace.json",
            "created_at": "2026-01-01T00:00:00+00:00",
            "query_result": {"query_summary": {"query_text": payload.get("query_text")}},
            "response": {"response_mode": "quote", "response_text": "Cats chase mice."},
            "learning": None,
            "runtime_episode": {"operation": "respond"},
            "dirty_state": False,
            "state_revision": 1,
        }

    @app.get("/terminus/living-loop")
    def living_loop() -> dict[str, Any]:
        return {
            "living_loop": {
                "runtime_episodes": [],
                "policy_decision": {
                    "schema_version": 1,
                    "action": "investigate_contradictions",
                    "recommendation": "Investigate contradictions.",
                    "reasons": [{"code": "contradicted_feedback", "detail": "test"}],
                    "risk": 0.75,
                    "expected_information_gain": 0.8,
                    "expected_goal_progress": 0.6,
                    "expected_cost": 0.35,
                    "uncertainty": 0.1,
                    "advisory": True,
                    "executable": False,
                    "target_episode_id": None,
                    "target_action_id": "act-1",
                    "action_id": "act-1",
                    "suggested_endpoint": "/terminus/living-loop",
                    "suggested_input": {},
                    "input": {},
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "feedback_summary": {
                    "feedback_count": 2,
                    "verified_count": 1,
                    "contradicted_count": 1,
                    "unverified_count": 0,
                    "recent_feedback": [{"target_type": "runtime_episode", "verdict": "contradicted"}],
                },
                "benchmark_telemetry": {
                    "endpoint_latency": {"feed": {"count": 1, "latency_ms_mean": 1.0}},
                    "runtime": {"tokens_per_second": 0.0},
                },
            },
            "dirty_state": False,
            "state_revision": 1,
            "token_count": 0,
        }

    @app.get("/terminus/policy-actuator")
    def policy_actuator() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "action": "investigate_contradictions",
            "recommendation": "Investigate contradictions.",
            "reasons": [{"code": "contradicted_feedback", "detail": "test"}],
            "risk": 0.75,
            "expected_information_gain": 0.8,
            "expected_goal_progress": 0.6,
            "expected_cost": 0.35,
            "uncertainty": 0.1,
            "advisory": True,
            "executable": False,
            "target_episode_id": None,
            "target_action_id": "act-1",
            "action_id": "act-1",
            "suggested_endpoint": "/terminus/living-loop",
            "suggested_input": {},
            "input": {},
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    @app.get("/terminus/replay-plan")
    def replay_plan(limit: int = 3) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "advisory": True,
            "executable": False,
            "endpoint": "/terminus/replay-plan",
            "limit": limit,
            "count": 1,
            "state_revision": 1,
            "token_count": 0,
            "snapshot_counts": {"runtime_episodes": 0, "actions": 0, "predictions": 0, "feedback": 2},
            "priority_rules_version": "deterministic-v1",
            "priority_weights": {"safety": 100.0},
            "plan_reason_codes": ["contradicted_feedback"],
            "candidates": [
                {
                    "candidate_id": "replay-1",
                    "rank": 1,
                    "target_type": "action",
                    "target_id": "act-1",
                    "target_ids": ["act-1"],
                    "operation": "workspace_search",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": "",
                    "reason_codes": ["contradicted_feedback"],
                    "priority_score": 125.0,
                    "priority_components": {"safety": 1.0},
                    "suggested_consolidation_action": "review_contradiction",
                    "suggested_endpoint": "/terminus/runtime-feedback",
                    "suggested_input": {"target_type": "action", "target_id": "act-1"},
                    "summary": "test",
                    "provenance": {},
                    "risk": 0.75,
                    "uncertainty": 0.1,
                    "latency": {},
                    "memory_health": {},
                    "feedback": {},
                    "policy": {},
                }
            ],
        }

    @app.get("/terminus/replay-sample/history")
    def replay_sample_history(limit: int = 3) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "endpoint": "/terminus/replay-sample/history",
            "count": 1,
            "limit": limit,
            "history": [
                {
                    "schema_version": 1,
                    "replay_sample_id": "replay-sample-1",
                    "execution_id": None,
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "mode": "sample",
                    "status": "recorded",
                    "reason": "operator-gated replay sample recorded without side effects",
                    "endpoint": "/terminus/replay-sample",
                    "operator_id": "benchmark-operator",
                    "operator_note": "",
                    "requested_candidate_id": "replay-1",
                    "target_type": "action",
                    "target_id": "act-1",
                    "requested_count": 1,
                    "alpha": 1.0,
                    "seed": None,
                    "candidate_ids": ["replay-1"],
                    "selected_candidate_ids": ["replay-1"],
                    "selected_candidates": [],
                    "safety_checks": {"passed": True},
                    "safety_flags": {"audit_only": True, "external_calls_made": False, "training_started": False},
                    "before": {"token_count": 0, "state_revision": 1, "action_history_count": 0, "feedback_count": 2},
                    "after": {"token_count": 0, "state_revision": 1, "action_history_count": 0, "feedback_count": 2},
                    "plan_summary": {"endpoint": "/terminus/replay-plan", "count": 1},
                }
            ],
        }

    @app.get("/terminus/runtime-traces/export")
    def export(limit: int = 3) -> dict[str, Any]:
        return {
            "export_kind": "terminus_runtime_trace_dataset_preview",
            "schema_version": 1,
            "training_role": "adapter_distillation_dataset_preview_only_not_training",
            "limit": limit,
            "count": 0,
            "replay_sample_summary": {
                "endpoint": "/terminus/replay-sample",
                "history_endpoint": "/terminus/replay-sample/history",
                "count": 1,
                "mode_counts": {"sample": 1},
                "status_counts": {"recorded": 1},
                "safety_flags": {"audit_only": True, "external_calls_made": False},
            },
            "examples": [],
        }

    @app.get("/terminus/replay-dataset/preview")
    def replay_dataset_preview(limit: int = 3) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "export_kind": "terminus_replay_dataset_preview",
            "training_role": "replay_dataset_preview_only_not_training_no_mutation",
            "created_at": "2026-01-01T00:00:00+00:00",
            "latest_export_timestamp": "2026-01-01T00:00:00+00:00",
            "latest_history_timestamp": "2026-01-01T00:00:01+00:00",
            "endpoint": "/terminus/replay-dataset/preview",
            "limit": limit,
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "provenance_counts": {},
            "example_type_counts": {},
            "safety_flags": {
                "preview_only": True,
                "training_started": False,
                "memory_mutated": False,
                "feedback_posted": False,
                "digital_action_executed": False,
                "external_calls_made": False,
            },
            "items": [],
            "empty_reason": "checkpoint_contains_no_eligible_sanitized_runtime_traces",
        }

    @app.post("/terminus/replay-dataset/bundle")
    def replay_dataset_bundle(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "export_kind": "terminus_replay_dataset_bundle_preview",
            "training_role": "replay_dataset_bundle_preview_only_not_training_operator_approved",
            "created_at": "2026-01-01T00:00:00+00:00",
            "endpoint": "/terminus/replay-dataset/bundle",
            "source_endpoint": "/terminus/replay-dataset/preview",
            "limit": int(payload.get("limit", 3)),
            "bundle_id": "terminus-replay-dataset-bundle-v1-empty",
            "bundle_version": "v1.empty",
            "source_count": 0,
            "count": 0,
            "excluded_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "preference_pair_count": 0,
            "sft_count": 0,
            "split_counts": {"train": 0, "holdout": 0, "eval": 0},
            "operator_approval": {
                "approved": True,
                "operator_id": payload.get("operator_id", "benchmark-operator"),
            },
            "training_gate": {
                "status": "blocked_preview_only",
                "eligible_for_training": False,
                "next_action": "run_offline_replay_training_eval_gate",
            },
            "safety_flags": {
                "preview_only": True,
                "training_started": False,
                "requires_separate_training_approval": True,
            },
            "empty_reason": "no_items_survived_bundle_packaging_gate",
        }

    @app.get("/terminus/replay-dataset/candidates")
    def replay_dataset_candidates(limit: int = 3) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "export_kind": "terminus_replay_dataset_candidates_preview",
            "training_role": "replay_dataset_preview_only_not_training_no_mutation",
            "created_at": "2026-01-01T00:00:00+00:00",
            "limit": limit,
            "count": 1,
            "candidates": [],
            "replay_plan_summary": {"endpoint": "/terminus/replay-plan", "count": 1},
            "safety_flags": {"preview_only": True, "external_calls_made": False},
        }

    @app.get("/terminus/replay-dataset/history")
    def replay_dataset_history(limit: int = 3) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "export_kind": "terminus_replay_dataset_history_preview",
            "training_role": "replay_dataset_preview_only_not_training_no_mutation",
            "created_at": "2026-01-01T00:00:00+00:00",
            "limit": limit,
            "count": 1,
            "source_endpoint": "/terminus/replay-sample/history",
            "history": [],
            "replay_sample_summary": {
                "endpoint": "/terminus/replay-sample",
                "count": 1,
                "latest_history_item": {"created_at": "2026-01-01T00:00:01+00:00"},
            },
            "safety_flags": {"preview_only": True, "external_calls_made": False},
        }

    return app


def test_benchmark_service_app_writes_json_shape_for_fake_app() -> None:
    root = _scratch_root("fake-app")
    try:
        output_path = root / "benchmark.json"
        result = benchmark_service_app(
            _fake_service_app(),
            output_path=output_path,
            checkpoint_path=root / "fake.pt",
        )

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        readme = output_path.parent / "README.md"
        assert loaded == result
        assert readme.exists()
        assert "HECSN Service Benchmark" in readme.read_text(encoding="utf-8")
        assert result["benchmark"] == "hecsn_service_endpoint_latency"
        assert result["schema_version"] == 1
        assert result["success"] is True
        assert result["total_latency_ms"] < 5000.0
        assert [item["name"] for item in result["endpoint_timings"]] == [
            "health",
            "status",
            "terminus",
            "feed",
            "query",
            "respond",
            "living_loop",
            "policy_actuator",
            "replay_plan",
            "replay_sample_history",
            "export",
            "replay_dataset_preview",
            "replay_dataset_bundle",
            "replay_dataset_candidates",
            "replay_dataset_history",
        ]
        for record in result["endpoint_timings"]:
            assert record["status_code"] == 200
            assert record["success"] is True
            assert record["latency_ms"] >= 0.0
        assert result["endpoints_by_name"]["export"]["path"] == "/terminus/runtime-traces/export"
        assert result["endpoints_by_name"]["status"]["path"] == "/status"
        assert result["endpoints_by_name"]["terminus"]["path"] == "/terminus"
        assert result["status_runtime_truth_summary"]["verdict"] == "alive"
        assert result["status_runtime_truth_summary"]["recommended_action"] == "continue_monitoring"
        assert result["status_runtime_truth_summary"]["evidence"]["token_count"] == 12
        assert result["terminus_runtime_truth_summary"]["verdict"] == "alive"
        assert result["living_loop_benchmark_telemetry"]["endpoint_latency"]["feed"]["count"] == 1
        assert result["feed_summary"]["tokens_processed"] > 0
        assert result["living_loop_benchmark_telemetry"]["feedback"]["feedback_count"] == 2
        assert result["feedback_telemetry"]["contradicted_count"] == 1
        assert result["endpoints_by_name"]["policy_actuator"]["path"] == "/terminus/policy-actuator"
        assert result["policy_actuator_summary"]["action"] == "investigate_contradictions"
        assert result["policy_actuator_summary"]["reason_codes"] == ["contradicted_feedback"]
        assert result["endpoints_by_name"]["replay_plan"]["path"] == "/terminus/replay-plan"
        assert result["endpoints_by_name"]["replay_sample_history"]["path"] == "/terminus/replay-sample/history"
        assert result["endpoints_by_name"]["replay_dataset_preview"]["path"] == "/terminus/replay-dataset/preview"
        assert result["endpoints_by_name"]["replay_dataset_bundle"]["path"] == "/terminus/replay-dataset/bundle"
        assert result["endpoints_by_name"]["replay_dataset_candidates"]["path"] == "/terminus/replay-dataset/candidates"
        assert result["endpoints_by_name"]["replay_dataset_history"]["path"] == "/terminus/replay-dataset/history"
        assert result["replay_plan_summary"]["endpoint"] == "/terminus/replay-plan"
        assert result["replay_plan_summary"]["top_candidate"]["suggested_consolidation_action"] == "review_contradiction"
        assert result["replay_sample_summary"]["endpoint"] == "/terminus/replay-sample"
        assert result["replay_sample_summary"]["history_endpoint"] == "/terminus/replay-sample/history"
        assert result["replay_sample_summary"]["count"] == 1
        assert result["replay_sample_summary"]["mode_counts"]["sample"] == 1
        assert result["replay_sample_summary"]["status_counts"]["recorded"] == 1
        assert result["replay_sample_summary"]["latest_history_item"]["selected_count"] == 1
        assert result["replay_sample_summary"]["safety_flags"]["audit_only"] is True
        assert result["replay_sample_summary"]["safety_flags"]["external_calls_made"] is False
        assert result["replay_executor_summary"]["count"] == 1
        assert result["trace_export_summary"]["count"] == 0
        assert result["replay_dataset_summary"]["export_kind"] == "terminus_replay_dataset_preview"
        assert result["replay_dataset_summary"]["training_role"] == "replay_dataset_preview_only_not_training_no_mutation"
        assert result["replay_dataset_summary"]["endpoint"] == "/terminus/replay-dataset/preview"
        assert result["replay_dataset_summary"]["positive_count"] == 0
        assert result["replay_dataset_summary"]["negative_count"] == 0
        assert result["replay_dataset_summary"]["empty_reason"] == "checkpoint_contains_no_eligible_sanitized_runtime_traces"
        assert result["replay_dataset_summary"]["latest_export_timestamp"] == "2026-01-01T00:00:00+00:00"
        assert result["replay_dataset_summary"]["latest_history_timestamp"] == "2026-01-01T00:00:01+00:00"
        assert result["replay_dataset_summary"]["safety_flags"]["external_calls_made"] is False
        assert result["replay_dataset_bundle_summary"]["export_kind"] == "terminus_replay_dataset_bundle_preview"
        assert result["replay_dataset_bundle_summary"]["training_role"] == "replay_dataset_bundle_preview_only_not_training_operator_approved"
        assert result["replay_dataset_bundle_summary"]["operator_approval"]["operator_id"] == "benchmark-operator"
        assert result["replay_dataset_bundle_summary"]["training_gate"]["status"] == "blocked_preview_only"
        assert result["replay_dataset_bundle_summary"]["training_gate"]["eligible_for_training"] is False
        assert result["replay_dataset_bundle_summary"]["safety_flags"]["training_started"] is False
        assert result["replay_dataset_bundle_summary"]["safety_flags"]["requires_separate_training_approval"] is True
        assert result["replay_dataset_candidates_summary"]["export_kind"] == "terminus_replay_dataset_candidates_preview"
        assert result["replay_dataset_history_summary"]["source_endpoint"] == "/terminus/replay-sample/history"
        assert result["replay_dataset_history_summary"]["latest_history_timestamp"] == "2026-01-01T00:00:01+00:00"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_service_benchmark_completes_with_tiny_checkpoint() -> None:
    root = _scratch_root("tiny-checkpoint")
    try:
        checkpoint_path = create_tiny_service_benchmark_checkpoint(root / "benchmark.pt")
        output_path = root / "result.json"

        result = run_service_benchmark(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            trace_dir=root / "traces",
            web_dist_dir=root / "missing-ui-dist",
            env_root=root,
            feed_text="Cats chase mice at night. Cats rest indoors during the day. " * 2,
            query_text="cats chase mice",
            top_k_candidates=4,
            top_k_memories=4,
            export_limit=2,
        )

        assert output_path.exists()
        assert (output_path.parent / "README.md").exists()
        assert result["success"] is True
        assert result["total_latency_ms"] < 30000.0
        assert result["endpoints_by_name"]["health"]["status_code"] == 200
        assert result["endpoints_by_name"]["status"]["status_code"] == 200
        assert result["endpoints_by_name"]["terminus"]["status_code"] == 200
        assert result["endpoints_by_name"]["feed"]["status_code"] == 200
        assert result["endpoints_by_name"]["query"]["status_code"] == 200
        assert result["endpoints_by_name"]["respond"]["status_code"] == 200
        assert result["endpoints_by_name"]["living_loop"]["status_code"] == 200
        assert result["endpoints_by_name"]["policy_actuator"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_plan"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_sample_history"]["status_code"] == 200
        assert result["endpoints_by_name"]["export"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_dataset_preview"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_dataset_bundle"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_dataset_candidates"]["status_code"] == 200
        assert result["endpoints_by_name"]["replay_dataset_history"]["status_code"] == 200
        assert isinstance(result["living_loop_benchmark_telemetry"], dict)
        assert isinstance(result["status_runtime_truth_summary"], dict)
        assert result["status_runtime_truth_summary"]["schema_version"] == 1
        assert result["status_runtime_truth_summary"]["verdict"] in {"alive", "degraded", "partial", "failed"}
        assert result["status_runtime_truth_summary"]["recommended_action"]
        assert isinstance(result["terminus_runtime_truth_summary"], dict)
        assert result["terminus_runtime_truth_summary"]["verdict"] == result["status_runtime_truth_summary"]["verdict"]
        assert isinstance(result["living_loop_benchmark_telemetry"]["feedback"], dict)
        assert result["feed_summary"]["feed_encoding_mode"] == "lexical_rolling_segments"
        assert result["feed_summary"]["concept_observation_mode"] == "sampled"
        assert result["feed_summary"]["concept_observations"] < result["feed_summary"]["tokens_processed"]
        assert isinstance(result["feedback_telemetry"], dict)
        assert isinstance(result["policy_actuator_summary"], dict)
        assert isinstance(result["policy_actuator_summary"]["action"], str)
        assert isinstance(result["replay_plan_summary"], dict)
        assert result["replay_plan_summary"]["endpoint"] == "/terminus/replay-plan"
        assert isinstance(result["replay_sample_summary"], dict)
        assert result["replay_sample_summary"]["endpoint"] == "/terminus/replay-sample"
        assert result["replay_sample_summary"]["history_endpoint"] == "/terminus/replay-sample/history"
        assert result["replay_sample_summary"]["safety_flags"]["audit_only"] is True
        assert result["trace_export_summary"]["count"] <= 2
        assert result["trace_export_summary"]["replay_dataset_summary"]["endpoint"] == "/terminus/replay-dataset/preview"
        assert result["replay_dataset_summary"]["count"] <= 2
        assert result["replay_dataset_summary"]["endpoint"] == "/terminus/replay-dataset/preview"
        assert "latest_export_timestamp" in result["replay_dataset_summary"]
        assert result["replay_dataset_summary"]["safety_flags"]["training_started"] is False
        assert result["replay_dataset_bundle_summary"]["export_kind"] == "terminus_replay_dataset_bundle_preview"
        assert result["replay_dataset_bundle_summary"]["operator_approval"]["approved"] is True
        assert result["replay_dataset_bundle_summary"]["training_gate"]["status"] == "blocked_preview_only"
        assert result["replay_dataset_bundle_summary"]["training_gate"]["eligible_for_training"] is False
        assert result["replay_dataset_bundle_summary"]["safety_flags"]["training_started"] is False
        assert result["replay_dataset_candidates_summary"]["count"] <= 2
        assert result["replay_dataset_history_summary"]["count"] >= 0
    finally:
        shutil.rmtree(root, ignore_errors=True)
