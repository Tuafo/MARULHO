from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request

import marulho.evaluation.service_benchmark as service_benchmark_module
from marulho.evaluation.service_benchmark import (
    benchmark_service_app,
    compare_service_benchmark_devices,
    compare_service_benchmark_against_accepted_baseline,
    compare_service_benchmark_report_files,
    compare_service_benchmark_reports,
    create_service_benchmark_accepted_baseline,
    create_tiny_service_benchmark_checkpoint,
    main as service_benchmark_main,
    run_service_benchmark_against_accepted_baseline,
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
            "memory_pressure": {"fill_fraction": 0.1, "size": 2, "capacity": 64, "pressure": "low"},
            "replay_role": "preview_only_not_training",
            "safety_flags": {"replay_dataset_preview_only": True, "training_started": False},
            "latency_ms": {"last_tick": 1.0, "tokens_per_second": 10.0},
            "evidence": {
                "configured": True,
                "running": True,
                "token_count": 12,
            },
        }

    def runtime_scope() -> dict[str, Any]:
        return {
            "device": {
                "requested_device": "auto",
                "env_device": None,
                "resolved_device": "cpu",
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_selected": False,
            },
            "cuda_first_runtime": {
                "enabled_when_available": True,
                "tensor_device": "cpu",
                "routing_search_device": "cpu",
                "routing_backend_cuda_capable": True,
                "unit_tests_default_cpu": True,
                "encoder_device_report": {"encoder": "rtf", "device": "cpu"},
                "subcortex_tensor_devices": {
                    "competitive": {"device": "cpu"},
                    "predictive": {"device": "cpu"},
                },
            },
        }

    @app.get("/status")
    def status() -> dict[str, Any]:
        return {
            "status": "ok",
            "token_count": 12,
            "runtime_truth": runtime_truth(),
            "runtime_scope": runtime_scope(),
        }

    @app.get("/terminus")
    def terminus() -> dict[str, Any]:
        return {
            "terminus_runtime": {
                "configured": True,
                "running": True,
                "ingestion": {
                    "warm_ready": True,
                    "full_warm_ready": True,
                    "ready_source_count": 1,
                    "full_queue_source_count": 1,
                    "total_buffered_tokens": 256,
                    "queue_target_tokens": 256,
                    "prewarm_last_duration_ms": 4.0,
                    "startup_warm_latency_ms": 6.0,
                },
            },
            "runtime_truth": runtime_truth(),
            "runtime_scope": runtime_scope(),
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


def _benchmark_report(
    *,
    verdict: str = "alive",
    success: bool = True,
    hot_p95: float = 100.0,
    hot_total: float = 240.0,
    configured: bool = True,
    tick_tokens: int = 24,
    hot_names: list[str] | None = None,
    device: str = "cpu",
    cuda_available: bool = False,
    observed_cuda_execution: bool = False,
) -> dict[str, Any]:
    hot_endpoint_names = hot_names or ["feed", "query", "respond"]
    cuda_selected = str(device).startswith("cuda")
    return {
        "schema_version": 1,
        "success": success,
        "endpoint_timings": [{"name": "status", "success": success, "latency_ms": 1.0}],
        "endpoint_metabolism_summary": {
            "setup": {"endpoint_names": ["terminus_configure", "terminus_tick"], "success": True},
            "hot_path": {
                "endpoint_names": hot_endpoint_names,
                "latency_ms_p95": hot_p95,
                "latency_ms_total": hot_total,
                "success": True,
            },
            "slow_path": {"endpoint_names": ["replay_plan", "replay_dataset_preview"], "success": True},
            "hot_path_budget": {"within_budget": hot_p95 <= 1000.0 and hot_total <= 3000.0},
        },
        "configured_source_summary": {
            "configured": configured,
            "source_name": "benchmark_local_source",
            "source_count": 1 if configured else 0,
            "tick_tokens_processed": tick_tokens,
            "not_hot_path": True,
        },
        "source_configuration_evidence": {
            "status": {
                "configured": configured,
                "source_count": 1 if configured else 0,
                "source_names": ["benchmark_local_source"] if configured else [],
            }
        },
        "status_runtime_truth_summary": {
            "schema_version": 1,
            "verdict": verdict,
            "recommended_action": "continue_monitoring" if verdict == "alive" else "configure_terminus_sources",
        },
        "runtime_device_evidence": {
            "status": {
                "summary_role": "observed_runtime_device_evidence_not_acceleration_claim",
                "requested_device": "auto",
                "env_device": device,
                "resolved_device": device,
                "cuda_available": cuda_available,
                "cuda_selected": cuda_selected,
                "cuda_device_count": 1 if cuda_available else 0,
                "tensor_device": device,
                "routing_search_device": device,
                "encoder_device": device,
                "observed_cuda_execution": observed_cuda_execution,
            },
            "terminus": {
                "summary_role": "observed_runtime_device_evidence_not_acceleration_claim",
                "resolved_device": device,
                "observed_cuda_execution": observed_cuda_execution,
            },
        },
    }


def test_service_benchmark_regression_gate_passes_for_configured_alive_run() -> None:
    comparison = compare_service_benchmark_reports(
        before_report=_benchmark_report(verdict="partial", hot_p95=120.0, hot_total=260.0),
        after_report=_benchmark_report(verdict="alive", hot_p95=130.0, hot_total=280.0),
    )

    assert comparison["status"] == "passed"
    assert comparison["runtime_truth"]["before"] == "partial"
    assert comparison["runtime_truth"]["after"] == "alive"
    assert comparison["checks"]["configured_source_alive"] is True
    assert comparison["checks"]["setup_not_in_hot_path"] is True
    assert comparison["claim_boundary"] == "regression_gate_only_no_runtime_mutation_no_cuda_speedup_claim"


def test_service_benchmark_regression_gate_fails_on_hot_path_regression() -> None:
    comparison = compare_service_benchmark_reports(
        before_report=_benchmark_report(verdict="alive", hot_p95=100.0, hot_total=200.0),
        after_report=_benchmark_report(verdict="alive", hot_p95=180.0, hot_total=400.0),
        hot_path_regression_tolerance=0.25,
    )

    assert comparison["status"] == "failed"
    assert comparison["checks"]["hot_path_p95_no_relative_regression"] is False
    assert comparison["checks"]["hot_path_total_no_relative_regression"] is False
    assert comparison["hot_path"]["allowed_after_p95_ms"] == 125.0
    assert comparison["hot_path"]["allowed_after_total_ms"] == 250.0


def test_service_benchmark_regression_gate_fails_when_setup_leaks_into_hot_path() -> None:
    comparison = compare_service_benchmark_reports(
        before_report=_benchmark_report(verdict="alive"),
        after_report=_benchmark_report(
            verdict="alive",
            hot_names=["feed", "query", "respond", "terminus_tick"],
        ),
    )

    assert comparison["status"] == "failed"
    assert comparison["checks"]["setup_not_in_hot_path"] is False
    assert comparison["endpoint_grouping"]["setup_leaked_into_hot_path"] is True


def test_service_benchmark_regression_gate_writes_json_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        before_path = root / "before.json"
        after_path = root / "after.json"
        output_path = root / "comparison.json"
        before_path.write_text(json.dumps(_benchmark_report(verdict="partial")), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")

        comparison = compare_service_benchmark_report_files(
            before_path=before_path,
            after_path=after_path,
            output_path=output_path,
        )
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == comparison
    assert comparison["artifact_kind"] == "marulho_service_benchmark_regression_gate"


def test_service_benchmark_cli_writes_regression_gate_report(capsys: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        before_path = root / "before.json"
        after_path = root / "after.json"
        output_path = root / "comparison.json"
        before_path.write_text(json.dumps(_benchmark_report(verdict="partial")), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")

        service_benchmark_main(
            [
                "--compare-before",
                str(before_path),
                "--compare-after",
                str(after_path),
                "--output",
                str(output_path),
            ]
        )
        stdout = capsys.readouterr().out
        comparison = json.loads(output_path.read_text(encoding="utf-8"))

    assert json.loads(stdout)["status"] == "passed"
    assert comparison["status"] == "passed"


def test_service_benchmark_device_comparison_reports_observed_cuda_delta() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_path = root / "device-comparison.json"
        cpu_report = _benchmark_report(
            verdict="alive",
            hot_p95=120.0,
            hot_total=300.0,
            device="cpu",
            cuda_available=True,
            observed_cuda_execution=False,
        )
        cuda_report = _benchmark_report(
            verdict="alive",
            hot_p95=90.0,
            hot_total=210.0,
            device="cuda",
            cuda_available=True,
            observed_cuda_execution=True,
        )

        comparison = compare_service_benchmark_devices(
            cpu_report=cpu_report,
            cuda_report=cuda_report,
            cpu_report_path=root / "cpu.json",
            cuda_report_path=root / "cuda.json",
            output_path=output_path,
        )
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == comparison
    assert comparison["artifact_kind"] == "marulho_service_benchmark_device_comparison"
    assert comparison["status"] == "passed"
    assert comparison["checks"]["cpu_observed_not_cuda"] is True
    assert comparison["checks"]["cuda_observed_execution"] is True
    assert comparison["endpoint_success_parity"]["parity_scope"] == "endpoint_success_names_only_not_semantic_output_equivalence"
    assert comparison["hot_path"]["p95_delta_ms_cuda_minus_cpu"] == -30.0
    assert comparison["hot_path"]["p95_cpu_over_cuda_ratio"] == 1.3333
    assert comparison["claim_boundary"].startswith("observed_cpu_cuda_device_and_latency_delta_only")


def test_service_benchmark_cli_writes_device_comparison_report(monkeypatch: Any, capsys: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint_path = root / "benchmark.pt"
        checkpoint_path.write_text("placeholder", encoding="utf-8")
        output_dir = root / "device-bundle"

        def fake_run_device_comparison(**kwargs: Any) -> dict[str, Any]:
            bundle_dir = Path(kwargs["output_dir"])
            bundle_dir.mkdir(parents=True, exist_ok=True)
            summary_path = bundle_dir / "device-comparison.json"
            summary = {
                "schema_version": 1,
                "artifact_kind": "marulho_service_benchmark_device_comparison",
                "status": "passed",
                "success": True,
                "paths": {"summary": str(summary_path)},
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return summary

        monkeypatch.setattr(
            service_benchmark_module,
            "run_service_benchmark_device_comparison",
            fake_run_device_comparison,
        )

        service_benchmark_main(
            [
                "--compare-devices",
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(output_dir),
            ]
        )
        stdout = capsys.readouterr().out
        assert (output_dir / "device-comparison.json").exists()

    assert json.loads(stdout)["status"] == "passed"


def test_service_benchmark_accepts_operator_reviewed_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report_path = root / "benchmark.json"
        baseline_path = root / "accepted-baseline.json"
        report_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")

        baseline = create_service_benchmark_accepted_baseline(
            report_path=report_path,
            output_path=baseline_path,
            accepted_by="operator-a",
            label="local-cpu-smoke",
            note="Accepted after configured-source smoke run.",
        )
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert loaded == baseline
    assert baseline["artifact_kind"] == "marulho_service_benchmark_accepted_baseline"
    assert baseline["status"] == "accepted"
    assert baseline["operator_review"]["accepted_by"] == "operator-a"
    assert baseline["operator_review"]["acceptance_hash_algorithm"] == "sha256_canonical_json"
    assert baseline["operator_review"]["acceptance_hash"] == service_benchmark_module._sha256_json(
        baseline["operator_review"]["acceptance_material"]
    )
    assert baseline["operator_review"]["acceptance_material"]["accepted_by"] == "operator-a"
    assert (
        baseline["operator_review"]["acceptance_material"]["source_report_sha256_canonical_json"]
        == baseline["source_report"]["sha256_canonical_json"]
    )
    assert baseline["label"] == "local-cpu-smoke"
    assert baseline["source_report"]["runtime_truth_verdict"] == "alive"
    assert baseline["baseline_id"].startswith("service-benchmark-baseline:")
    assert baseline["claim_boundary"] == "accepted_baseline_manifest_only_no_runtime_mutation_no_cuda_speedup_claim"


def test_service_benchmark_refuses_unreviewed_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report_path = root / "benchmark.json"
        report_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")

        try:
            create_service_benchmark_accepted_baseline(
                report_path=report_path,
                output_path=root / "accepted-baseline.json",
                accepted_by="",
            )
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure branch
            raise AssertionError("Expected baseline acceptance to require operator identity")

    assert "accepted_by_present" in message


def test_service_benchmark_compares_after_run_against_accepted_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        before_path = root / "before.json"
        baseline_path = root / "accepted-baseline.json"
        after_path = root / "after.json"
        output_path = root / "comparison.json"
        before_path.write_text(json.dumps(_benchmark_report(verdict="partial", hot_p95=120.0, hot_total=260.0)), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive", hot_p95=130.0, hot_total=280.0)), encoding="utf-8")
        baseline = create_service_benchmark_accepted_baseline(
            report_path=before_path,
            output_path=baseline_path,
            accepted_by="operator-a",
            label="baseline-a",
        )

        comparison = compare_service_benchmark_against_accepted_baseline(
            baseline_path=baseline_path,
            after_path=after_path,
            output_path=output_path,
        )
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == comparison
    assert comparison["status"] == "passed"
    assert comparison["runtime_truth"]["before"] == "partial"
    assert comparison["runtime_truth"]["after"] == "alive"
    assert comparison["accepted_baseline"]["baseline_id"] == baseline["baseline_id"]
    assert comparison["accepted_baseline"]["accepted_by"] == "operator-a"
    assert comparison["accepted_baseline"]["label"] == "baseline-a"
    assert comparison["accepted_baseline"]["claim_boundary"] == "accepted_baseline_used_for_report_only_regression_gate"


def test_service_benchmark_baseline_compare_rejects_tampered_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report_path = root / "benchmark.json"
        baseline_path = root / "accepted-baseline.json"
        after_path = root / "after.json"
        report_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")
        baseline = create_service_benchmark_accepted_baseline(
            report_path=report_path,
            output_path=baseline_path,
            accepted_by="operator-a",
        )
        baseline["baseline_report_snapshot"]["endpoint_metabolism_summary"]["hot_path"]["latency_ms_p95"] = 1.0
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        try:
            compare_service_benchmark_against_accepted_baseline(
                baseline_path=baseline_path,
                after_path=after_path,
            )
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure branch
            raise AssertionError("Expected tampered baseline snapshot to be rejected")

    assert "hash does not match" in message


def test_service_benchmark_baseline_compare_rejects_tampered_operator_acceptance() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        report_path = root / "benchmark.json"
        baseline_path = root / "accepted-baseline.json"
        after_path = root / "after.json"
        report_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")
        baseline = create_service_benchmark_accepted_baseline(
            report_path=report_path,
            output_path=baseline_path,
            accepted_by="operator-a",
        )
        baseline["operator_review"]["acceptance_material"]["accepted_by"] = "operator-b"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        try:
            compare_service_benchmark_against_accepted_baseline(
                baseline_path=baseline_path,
                after_path=after_path,
            )
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure branch
            raise AssertionError("Expected tampered operator acceptance to be rejected")

    assert "operator acceptance hash" in message


def test_service_benchmark_cli_accepts_baseline_and_compares_against_it(capsys: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        before_path = root / "before.json"
        after_path = root / "after.json"
        baseline_path = root / "accepted-baseline.json"
        comparison_path = root / "comparison.json"
        before_path.write_text(json.dumps(_benchmark_report(verdict="partial")), encoding="utf-8")
        after_path.write_text(json.dumps(_benchmark_report(verdict="alive")), encoding="utf-8")

        service_benchmark_main(
            [
                "--accept-baseline-from",
                str(before_path),
                "--accepted-by",
                "operator-a",
                "--baseline-label",
                "cli-baseline",
                "--output",
                str(baseline_path),
            ]
        )
        baseline_stdout = json.loads(capsys.readouterr().out)
        service_benchmark_main(
            [
                "--compare-baseline",
                str(baseline_path),
                "--compare-after",
                str(after_path),
                "--output",
                str(comparison_path),
            ]
        )
        comparison_stdout = json.loads(capsys.readouterr().out)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert baseline_stdout["status"] == "accepted"
    assert baseline_stdout["baseline_id"] == baseline["baseline_id"]
    assert comparison_stdout["status"] == "passed"
    assert comparison["accepted_baseline"]["baseline_id"] == baseline["baseline_id"]


def test_service_benchmark_cli_runs_fresh_bundle_against_baseline(capsys: Any, monkeypatch: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        baseline_source_path = root / "baseline-source.json"
        baseline_path = root / "accepted-baseline.json"
        bundle_dir = root / "bundle"
        baseline_source_path.write_text(
            json.dumps(_benchmark_report(verdict="alive", hot_p95=500.0, hot_total=900.0)),
            encoding="utf-8",
        )
        baseline = create_service_benchmark_accepted_baseline(
            report_path=baseline_source_path,
            output_path=baseline_path,
            accepted_by="operator-a",
        )
        after_report = _benchmark_report(verdict="alive", hot_p95=520.0, hot_total=940.0)
        captured: dict[str, Any] = {}

        def fake_run_service_benchmark(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            output_path = Path(kwargs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(after_report)
            payload["output_path"] = str(output_path)
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload

        monkeypatch.setattr(service_benchmark_module, "run_service_benchmark", fake_run_service_benchmark)

        service_benchmark_main(
            [
                "--run-against-baseline",
                str(baseline_path),
                "--checkpoint",
                str(root / "tiny.pt"),
                "--output",
                str(bundle_dir),
                "--configure-local-source",
                "--local-source-tick-steps",
                "1",
                "--local-source-tick-tokens",
                "128",
                "--local-source-queue-target-tokens",
                "256",
            ]
        )
        stdout = json.loads(capsys.readouterr().out)
        summary = json.loads((bundle_dir / "bundle-summary.json").read_text(encoding="utf-8"))
        comparison = json.loads((bundle_dir / "comparison.json").read_text(encoding="utf-8"))
        fresh_benchmark_exists = (bundle_dir / "fresh-benchmark.json").exists()
        comparison_exists = (bundle_dir / "comparison.json").exists()

    assert stdout["status"] == "passed"
    assert stdout["success"] is True
    assert fresh_benchmark_exists
    assert comparison_exists
    assert summary["artifact_kind"] == "marulho_service_benchmark_baseline_run_bundle"
    assert summary["accepted_baseline"]["baseline_id"] == baseline["baseline_id"]
    assert comparison["accepted_baseline"]["baseline_id"] == baseline["baseline_id"]
    assert captured["configure_local_source"] is True
    assert captured["local_source_tick_steps"] == 1
    assert captured["local_source_tick_tokens"] == 128
    assert captured["local_source_queue_target_tokens"] == 256
    assert captured["local_source_prewarm_on_startup"] is True
    assert captured["local_source_prewarm_wait_seconds"] == 5.0
    assert summary["claim_boundary"] == "fresh_benchmark_plus_baseline_compare_slow_path_no_runtime_mutation_no_cuda_speedup_claim"


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
        assert "MARULHO Service Benchmark" in readme.read_text(encoding="utf-8")
        assert result["benchmark"] == "marulho_service_endpoint_latency"
        assert result["schema_version"] == 1
        assert result["success"] is True
        assert result["trainer_stage_profile"] is None
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
        metabolism = result["endpoint_metabolism_summary"]
        assert metabolism["setup"]["count"] == 0
        assert metabolism["hot_path"]["endpoint_names"] == ["feed", "query", "respond"]
        assert metabolism["hot_path"]["count"] == 3
        assert metabolism["hot_path"]["success"] is True
        assert metabolism["hot_path"]["latency_ms_p95"] is not None
        assert metabolism["hot_path_budget"]["within_budget"] is True
        assert metabolism["hot_path_budget"]["hot_path_protection_role"] == "benchmark_evidence_only_not_runtime_work"
        assert "replay_dataset_bundle" in metabolism["slow_path"]["endpoint_names"]
        assert metabolism["uncategorized"]["count"] == 0
        assert metabolism["semantics"]["setup_endpoints"] == ["terminus_configure", "terminus_tick"]
        assert metabolism["semantics"]["hot_path_endpoints"] == ["feed", "query", "respond"]
        assert result["configured_source_summary"] is None
        device_evidence = result["runtime_device_evidence"]
        assert device_evidence["semantics"]["claim_boundary"] == "observed_device_placement_only_not_cuda_speedup"
        assert device_evidence["status"]["summary_role"] == "observed_runtime_device_evidence_not_acceleration_claim"
        assert device_evidence["status"]["tensor_device"] == "cpu"
        assert device_evidence["status"]["routing_search_device"] == "cpu"
        assert device_evidence["status"]["encoder_device"] == "cpu"
        assert device_evidence["status"]["observed_cuda_execution"] is False
        assert device_evidence["status"]["cuda_fallback_reason"] == "cuda_not_available"
        assert device_evidence["status"]["subcortex_device_sections"] == ["competitive", "predictive"]
        assert device_evidence["terminus"]["resolved_device"] == "cpu"
        assert result["endpoints_by_name"]["export"]["path"] == "/terminus/runtime-traces/export"
        assert result["endpoints_by_name"]["status"]["path"] == "/status"
        assert result["endpoints_by_name"]["terminus"]["path"] == "/terminus"
        assert result["status_runtime_truth_summary"]["verdict"] == "alive"
        assert result["status_runtime_truth_summary"]["recommended_action"] == "continue_monitoring"
        assert result["status_runtime_truth_summary"]["evidence"]["token_count"] == 12
        assert result["terminus_runtime_truth_summary"]["verdict"] == "alive"
        assert result["source_configuration_evidence"]["terminus"]["configured"] is True
        assert "Long-test calls quick_start_terminus" in result["source_configuration_evidence"]["semantics"]["long_test_difference"]
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
            configure_local_source=True,
            local_source_tick_steps=1,
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
        metabolism = result["endpoint_metabolism_summary"]
        assert metabolism["setup"]["endpoint_names"] == ["terminus_configure", "terminus_tick"]
        assert metabolism["setup"]["success"] is True
        assert metabolism["hot_path"]["endpoint_names"] == ["feed", "query", "respond"]
        assert metabolism["hot_path"]["count"] == 3
        assert metabolism["hot_path"]["success"] is True
        assert metabolism["hot_path"]["latency_ms_total"] <= result["total_latency_ms"]
        assert metabolism["hot_path_budget"]["within_budget"] is True
        assert metabolism["slow_path"]["count"] == 7
        assert metabolism["semantics"]["setup_note"].startswith("Configuration and manual tick")
        assert metabolism["semantics"]["slow_path_note"].startswith("Replay, export, bundle")
        configured_source = result["configured_source_summary"]
        assert configured_source["enabled"] is True
        assert configured_source["configured"] is True
        assert configured_source["source_name"] == "benchmark_local_source"
        assert configured_source["source_count"] == 1
        assert configured_source["tick_steps"] == 1
        assert configured_source["tick_tokens_processed"] > 0
        assert configured_source["background_tokens_processed"] > 0
        assert configured_source["not_hot_path"] is True
        device_evidence = result["runtime_device_evidence"]
        assert device_evidence["semantics"]["cuda_claim_requires"].startswith("observed_cuda_execution_true")
        assert isinstance(device_evidence["status"], dict)
        assert isinstance(device_evidence["terminus"], dict)
        assert device_evidence["status"]["summary_role"] == "observed_runtime_device_evidence_not_acceleration_claim"
        assert device_evidence["status"]["tensor_device"]
        assert device_evidence["status"]["encoder_device"]
        assert device_evidence["status"]["observed_cuda_execution"] is (
            str(device_evidence["status"]["tensor_device"]).startswith("cuda")
            or str(device_evidence["status"]["routing_search_device"]).startswith("cuda")
            or str(device_evidence["status"]["encoder_device"]).startswith("cuda")
        )
        assert isinstance(result["living_loop_benchmark_telemetry"], dict)
        assert isinstance(result["status_runtime_truth_summary"], dict)
        assert result["status_runtime_truth_summary"]["schema_version"] == 1
        assert result["status_runtime_truth_summary"]["verdict"] in {"alive", "degraded", "partial", "failed"}
        assert result["status_runtime_truth_summary"]["recommended_action"]
        assert isinstance(result["source_configuration_evidence"], dict)
        assert "semantics" in result["source_configuration_evidence"]
        assert result["source_configuration_evidence"]["status"]["configured"] is True
        assert result["source_configuration_evidence"]["status"]["source_count"] == 1
        assert result["source_configuration_evidence"]["status"]["source_names"] == ["benchmark_local_source"]
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


def test_benchmark_service_app_reports_unavailable_trainer_profile_for_fake_app() -> None:
    root = _scratch_root("fake-app-trainer-profile")
    try:
        result = benchmark_service_app(
            _fake_service_app(),
            output_path=root / "benchmark.json",
            checkpoint_path=root / "fake.pt",
            profile_trainer_stages=True,
        )

        profile = result["trainer_stage_profile"]
        assert profile["enabled"] is False
        assert profile["count"] == 0
        assert profile["unavailable_reason"] == "app_state_marulho_manager_trainer_missing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_service_benchmark_preserves_unchanged_local_source_mtime(monkeypatch) -> None:
    root = _scratch_root("local-source-cache")
    try:
        output_path = root / "result.json"
        source_path = root / "benchmark-local-source.txt"
        source_path.write_text("stable local source", encoding="utf-8")
        original_mtime = source_path.stat().st_mtime_ns

        def fake_benchmark_service_app(_app: FastAPI, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["configured_source_path"] == source_path
            return {"output_path": str(output_path), "success": True}

        monkeypatch.setattr(service_benchmark_module, "create_app", lambda **_: _fake_service_app())
        monkeypatch.setattr(service_benchmark_module, "benchmark_service_app", fake_benchmark_service_app)

        result = run_service_benchmark(
            checkpoint_path=root / "checkpoint.pt",
            output_path=output_path,
            configure_local_source=True,
            local_source_text="stable local source",
        )

        assert result["success"] is True
        assert source_path.stat().st_mtime_ns == original_mtime
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_benchmark_configured_source_summary_includes_tick_concept_observation() -> None:
    app = _fake_service_app()

    @app.post("/terminus/configure")
    def terminus_configure() -> dict[str, Any]:
        return {
            "terminus_runtime": {
                "configured": True,
                "source_count": 1,
            },
        }

    @app.post("/terminus/tick")
    def terminus_tick() -> dict[str, Any]:
        return {
            "terminus_runtime": {
                "last_tick_token_delta": 24,
                "background_tokens_processed": 24,
                "last_tick_duration_ms": 12.5,
                "source_progress": [
                    {
                        "cache_write_count": 1,
                        "cache_schedule_count": 3,
                        "cache_skip_count": 2,
                        "cache_failure_count": 0,
                        "cache_pending": True,
                        "last_cache_update_mode": "skipped_unchanged_material",
                    }
                ],
            },
            "tick_summaries": [
                {
                    "source": {
                        "concept_observation": {
                            "mode": "sampled_batched",
                            "attempts": 3,
                            "observations": 3,
                            "batches": 1,
                            "structural_maintenance_passes": 1,
                        }
                    }
                }
            ],
        }

    root = _scratch_root("configured-source-concept-summary")
    try:
        source_path = root / "source.txt"
        source_path.write_text("source text", encoding="utf-8")
        result = benchmark_service_app(
            app,
            output_path=root / "result.json",
            configured_source_path=source_path,
            configured_source_tick_steps=1,
            feed_text="cats",
            query_text="cats",
        )

        concept_observation = result["configured_source_summary"]["concept_observation"]
        assert concept_observation["mode"] == "sampled_batched"
        assert concept_observation["attempts"] == 3
        assert concept_observation["structural_maintenance_passes"] == 1
        source_cache = result["configured_source_summary"]["source_cache"]
        assert source_cache["cache_write_count"] == 1
        assert source_cache["cache_schedule_count"] == 3
        assert source_cache["cache_skip_count"] == 2
        assert source_cache["cache_failure_count"] == 0
        assert source_cache["cache_pending"] is True
        assert source_cache["last_cache_update_mode"] == "skipped_unchanged_material"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_benchmark_service_app_configures_explicit_source_tick_window() -> None:
    app = _fake_service_app()
    captured: dict[str, Any] = {}

    @app.post("/terminus/configure")
    async def terminus_configure(request: Request) -> dict[str, Any]:
        captured.update(await request.json())
        return {
            "terminus_runtime": {
                "configured": True,
                "source_count": 1,
            },
        }

    @app.post("/terminus/tick")
    def terminus_tick() -> dict[str, Any]:
        return {
            "terminus_runtime": {
                "last_tick_token_delta": 128,
                "background_tokens_processed": 128,
                "last_tick_duration_ms": 24.0,
                "source_progress": [],
            }
        }

    root = _scratch_root("configured-source-tick-window")
    try:
        source_path = root / "source.txt"
        source_path.write_text("source text", encoding="utf-8")
        result = benchmark_service_app(
            app,
            output_path=root / "result.json",
            configured_source_path=source_path,
            configured_source_tick_steps=1,
            configured_source_tick_tokens=128,
            configured_source_queue_target_tokens=256,
            feed_text="cats",
            query_text="cats",
        )

        assert captured["tick_tokens"] == 128
        assert captured["ingestion"]["queue_target_tokens"] == 256
        assert captured["ingestion"]["prewarm_on_startup"] is True
        assert result["configured_source_summary"]["tick_tokens_processed"] == 128
        warmup = result["configured_source_summary"]["warmup"]
        assert warmup["enabled"] is True
        assert warmup["mode"] == "prewarm_before_measured_tick"
        assert warmup["not_hot_path"] is True
        assert warmup["full_warm_ready"] is True
        assert warmup["total_buffered_tokens"] == 256
    finally:
        shutil.rmtree(root, ignore_errors=True)
