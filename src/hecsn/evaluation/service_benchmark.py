from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.readme_reports import write_json_report_with_readme
from hecsn.service.api import create_app
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


SERVICE_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_FEED_TEXT = "Cats chase mice at night. Cats rest indoors during the day. " * 4
DEFAULT_QUERY_TEXT = "cats chase mice"


def create_tiny_service_benchmark_checkpoint(path: str | Path) -> Path:
    """Create a tiny deterministic checkpoint for local service smoke benchmarks."""
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
    return save_trainer_checkpoint(
        path,
        trainer,
        metadata={"benchmark": "service_benchmark", "synthetic": True},
    )


def _response_json(response: Any) -> Any:
    content_type = str(response.headers.get("content-type", ""))
    if "application/json" not in content_type:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _measure_endpoint(
    client: TestClient,
    *,
    name: str,
    method: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    started = time.perf_counter()
    response = None
    error: BaseException | None = None
    try:
        response = client.request(method, path, json=json_body, params=params)
        body = _response_json(response)
    except Exception as exc:  # pragma: no cover - exercised through caller error paths
        error = exc
        body = None
    latency_ms = (time.perf_counter() - started) * 1000.0

    status_code = None if response is None else int(response.status_code)
    success = error is None and status_code is not None and 200 <= status_code < 400
    record: dict[str, Any] = {
        "name": name,
        "method": method,
        "path": path,
        "latency_ms": round(float(latency_ms), 3),
        "success": bool(success),
        "status_code": status_code,
    }
    if params:
        record["params"] = dict(params)
    if response is not None:
        record["response_size_bytes"] = int(len(response.content))
        if isinstance(body, dict):
            record["response_json_keys"] = sorted(str(key) for key in body)
    if error is not None:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    return record, body


def _write_json(output_path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json_report_with_readme(
        output_path,
        payload,
        title="HECSN Service Benchmark",
    )


def _latest_replay_dataset_history_timestamp(history_body: Any) -> str | None:
    if not isinstance(history_body, dict):
        return None
    history = history_body.get("history")
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and item.get("created_at"):
                return str(item["created_at"])
    replay_sample_summary = history_body.get("replay_sample_summary")
    if isinstance(replay_sample_summary, dict):
        latest = replay_sample_summary.get("latest_history_item")
        if isinstance(latest, dict) and latest.get("created_at"):
            return str(latest["created_at"])
    return None


def _summarize_runtime_truth(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    runtime_truth = body.get("runtime_truth")
    if not isinstance(runtime_truth, dict):
        return None
    return {
        key: runtime_truth.get(key)
        for key in (
            "schema_version",
            "generated_at",
            "verdict",
            "recommended_action",
            "cortex_available",
            "source_configuration",
            "memory_pressure",
            "replay_role",
            "safety_flags",
            "latency_ms",
            "evidence",
        )
        if key in runtime_truth
    }


def _summarize_source_configuration(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    runtime_truth = body.get("runtime_truth")
    if isinstance(runtime_truth, dict) and isinstance(runtime_truth.get("source_configuration"), dict):
        return dict(runtime_truth["source_configuration"])
    terminus_runtime = body.get("terminus_runtime")
    if not isinstance(terminus_runtime, dict):
        return None
    source_bank = [dict(item) for item in list(terminus_runtime.get("source_bank") or []) if isinstance(item, dict)]
    ingestion = terminus_runtime.get("ingestion") if isinstance(terminus_runtime.get("ingestion"), dict) else {}
    return {
        "configured": bool(terminus_runtime.get("configured")),
        "source_count": int(len(source_bank)),
        "source_names": [str(item.get("name", "")) for item in source_bank],
        "source_types": [str(item.get("source_type", "auto")) for item in source_bank],
        "tick_tokens": int(terminus_runtime.get("tick_tokens", 0) or 0),
        "repeat_sources": bool(terminus_runtime.get("repeat_sources", True)),
        "ingestion": dict(ingestion),
    }


def _summarize_replay_sample_history(history_body: Any, fallback: Any = None) -> dict[str, Any] | None:
    if isinstance(history_body, dict):
        history = history_body.get("history")
    else:
        history = None
    records = [dict(item) for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    if not records and isinstance(fallback, dict):
        existing = fallback.get("replay_sample_summary") or fallback.get("replay_executor_summary")
        if isinstance(existing, dict):
            return dict(existing)
    if not records and not isinstance(history_body, dict):
        return None

    mode_counts: Counter[str] = Counter({"dry_run": 0, "sample": 0, "execute": 0})
    status_counts: Counter[str] = Counter()
    selected_count = 0
    for record in records:
        mode = str(record.get("mode") or "sample")
        if mode not in {"dry_run", "sample", "execute"}:
            mode = "sample"
        status = str(record.get("status") or "recorded")
        mode_counts[mode] += 1
        status_counts[status] += 1
        selected_ids = record.get("selected_candidate_ids")
        if isinstance(selected_ids, list):
            selected_count += len(selected_ids)
        elif isinstance(record.get("selected_candidates"), list):
            selected_count += len(record["selected_candidates"])

    latest = records[0] if records else {}
    latest_selected_ids = latest.get("selected_candidate_ids") if isinstance(latest.get("selected_candidate_ids"), list) else []
    latest_safety_flags = {
        "audit_only": True,
        "operator_confirmed": False,
        "training_started": False,
        "sleep_started": False,
        "memory_verification_promoted": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
        "memory_mutated": False,
        "state_revision_mutated": False,
        "token_count_mutated": False,
        "action_history_mutated": False,
        "feedback_mutated": False,
        "not_promoted": True,
    }
    if isinstance(latest.get("safety_flags"), dict):
        latest_safety_flags.update(latest["safety_flags"])
    latest_item = None
    if latest:
        latest_item = {
            "schema_version": latest.get("schema_version", 1),
            "replay_sample_id": latest.get("replay_sample_id"),
            "execution_id": latest.get("execution_id"),
            "created_at": latest.get("created_at"),
            "mode": latest.get("mode"),
            "status": latest.get("status"),
            "endpoint": latest.get("endpoint", "/terminus/replay-sample"),
            "target_type": latest.get("target_type"),
            "target_id": latest.get("target_id"),
            "selected_count": len(latest_selected_ids),
            "selected_candidate_ids": latest_selected_ids[:20],
            "safety_flags": dict(latest_safety_flags),
        }
    history_count = int((history_body or {}).get("count", len(records)) or 0) if isinstance(history_body, dict) else len(records)
    return {
        "schema_version": 1,
        "endpoint": "/terminus/replay-sample",
        "execution_endpoint": "/terminus/replay-execute",
        "history_endpoint": "/terminus/replay-sample/history",
        "execution_history_endpoint": "/terminus/replay-execute/history",
        "count": history_count,
        "history_count": history_count,
        "selected_count": int(selected_count),
        "latest_selected_count": int(len(latest_selected_ids)),
        "mode_counts": dict(mode_counts),
        "status_counts": dict(status_counts),
        "latest_history_item": latest_item,
        "safety_flags": dict(latest_safety_flags),
        "audit_only": True,
        "advisory": True,
        "executable": False,
    }


def benchmark_service_app(
    app: FastAPI,
    *,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    feed_text: str = DEFAULT_FEED_TEXT,
    query_text: str = DEFAULT_QUERY_TEXT,
    top_k_candidates: int = 5,
    top_k_memories: int = 5,
    top_chars: int = 6,
    export_limit: int = 3,
    include_living_loop_telemetry: bool = True,
) -> dict[str, Any]:
    """Measure the local service endpoints in-process and write JSON results."""
    started = time.perf_counter()
    endpoint_timings: list[dict[str, Any]] = []
    response_bodies: dict[str, Any] = {}

    with TestClient(app) as client:
        requests: tuple[dict[str, Any], ...] = (
            {"name": "health", "method": "GET", "path": "/health"},
            {"name": "status", "method": "GET", "path": "/status"},
            {"name": "terminus", "method": "GET", "path": "/terminus"},
            {
                "name": "feed",
                "method": "POST",
                "path": "/feed",
                "json_body": {"text": feed_text},
            },
            {
                "name": "query",
                "method": "POST",
                "path": "/query",
                "json_body": {
                    "query_text": query_text,
                    "top_k_candidates": int(top_k_candidates),
                    "top_k_memories": int(top_k_memories),
                    "top_chars": int(top_chars),
                },
            },
            {
                "name": "respond",
                "method": "POST",
                "path": "/respond",
                "json_body": {
                    "query_text": query_text,
                    "top_k_candidates": int(top_k_candidates),
                    "top_k_memories": int(top_k_memories),
                    "top_chars": int(top_chars),
                    "max_evidence_items": 3,
                    "learn_mode": "none",
                },
            },
            {"name": "living_loop", "method": "GET", "path": "/terminus/living-loop"},
            {"name": "policy_actuator", "method": "GET", "path": "/terminus/policy-actuator"},
            {"name": "replay_plan", "method": "GET", "path": "/terminus/replay-plan", "params": {"limit": int(export_limit)}},
            {"name": "replay_sample_history", "method": "GET", "path": "/terminus/replay-sample/history", "params": {"limit": int(export_limit)}},
            {
                "name": "export",
                "method": "GET",
                "path": "/terminus/runtime-traces/export",
                "params": {"limit": int(export_limit)},
            },
            {
                "name": "replay_dataset_preview",
                "method": "GET",
                "path": "/terminus/replay-dataset/preview",
                "params": {"limit": int(export_limit)},
            },
            {
                "name": "replay_dataset_bundle",
                "method": "POST",
                "path": "/terminus/replay-dataset/bundle",
                "json_body": {
                    "operator_id": "benchmark-operator",
                    "operator_note": "Benchmark package preview only.",
                    "confirmation": True,
                    "limit": int(export_limit),
                    "holdout_fraction": 0.2,
                    "eval_fraction": 0.2,
                    "seed": 17,
                },
            },
            {
                "name": "replay_dataset_candidates",
                "method": "GET",
                "path": "/terminus/replay-dataset/candidates",
                "params": {"limit": int(export_limit)},
            },
            {
                "name": "replay_dataset_history",
                "method": "GET",
                "path": "/terminus/replay-dataset/history",
                "params": {"limit": int(export_limit)},
            },
        )
        for request in requests:
            record, body = _measure_endpoint(client, **request)
            endpoint_timings.append(record)
            response_bodies[str(request["name"])] = body

    living_loop_telemetry: dict[str, Any] | None = None
    feedback_telemetry: dict[str, Any] | None = None
    status_runtime_truth_summary = _summarize_runtime_truth(response_bodies.get("status"))
    terminus_runtime_truth_summary = _summarize_runtime_truth(response_bodies.get("terminus"))
    status_source_configuration = _summarize_source_configuration(response_bodies.get("status"))
    terminus_source_configuration = _summarize_source_configuration(response_bodies.get("terminus"))
    living_body = response_bodies.get("living_loop")
    if include_living_loop_telemetry and isinstance(living_body, dict):
        living_loop = living_body.get("living_loop")
        if isinstance(living_loop, dict) and isinstance(living_loop.get("benchmark_telemetry"), dict):
            living_loop_telemetry = dict(living_loop["benchmark_telemetry"])
            if isinstance(living_loop_telemetry.get("feedback"), dict):
                feedback_telemetry = dict(living_loop_telemetry["feedback"])
            elif isinstance(living_loop.get("feedback_summary"), dict):
                feedback_telemetry = dict(living_loop["feedback_summary"])
                living_loop_telemetry["feedback"] = feedback_telemetry
        elif isinstance(living_loop, dict) and isinstance(living_loop.get("feedback_summary"), dict):
            feedback_telemetry = dict(living_loop["feedback_summary"])
            living_loop_telemetry = {"feedback": feedback_telemetry}

    export_summary: dict[str, Any] | None = None
    export_body = response_bodies.get("export")
    if isinstance(export_body, dict):
        export_summary = {
            key: export_body.get(key)
            for key in (
                "export_kind",
                "schema_version",
                "training_role",
                "limit",
                "count",
                "endpoint",
                "replay_sample_summary",
                "replay_executor_summary",
                "replay_dataset_summary",
            )
            if key in export_body
        }

    replay_dataset_summary: dict[str, Any] | None = None
    replay_dataset_body = response_bodies.get("replay_dataset_preview")
    if isinstance(replay_dataset_body, dict):
        replay_dataset_summary = {
            key: replay_dataset_body.get(key)
            for key in (
                "export_kind",
                "schema_version",
                "training_role",
                "created_at",
                "latest_export_timestamp",
                "latest_history_timestamp",
                "endpoint",
                "filter_endpoint",
                "limit",
                "max_limit",
                "count",
                "positive_count",
                "negative_count",
                "provenance_counts",
                "example_type_counts",
                "safety_flags",
                "empty_reason",
            )
            if key in replay_dataset_body
        }
        replay_dataset_summary.setdefault(
            "latest_export_timestamp",
            replay_dataset_body.get("created_at"),
        )
        latest_history_timestamp = _latest_replay_dataset_history_timestamp(
            response_bodies.get("replay_dataset_history")
        )
        if latest_history_timestamp is not None:
            replay_dataset_summary["latest_history_timestamp"] = latest_history_timestamp

    replay_dataset_candidates_summary: dict[str, Any] | None = None
    replay_dataset_candidates_body = response_bodies.get("replay_dataset_candidates")
    if isinstance(replay_dataset_candidates_body, dict):
        replay_dataset_candidates_summary = {
            key: replay_dataset_candidates_body.get(key)
            for key in ("export_kind", "schema_version", "training_role", "limit", "count", "replay_plan_summary", "safety_flags")
            if key in replay_dataset_candidates_body
        }

    replay_dataset_bundle_summary: dict[str, Any] | None = None
    replay_dataset_bundle_body = response_bodies.get("replay_dataset_bundle")
    if isinstance(replay_dataset_bundle_body, dict):
        replay_dataset_bundle_summary = {
            key: replay_dataset_bundle_body.get(key)
            for key in (
                "export_kind",
                "schema_version",
                "training_role",
                "bundle_id",
                "bundle_version",
                "source_count",
                "count",
                "excluded_count",
                "positive_count",
                "negative_count",
                "preference_pair_count",
                "sft_count",
                "split_counts",
                "operator_approval",
                "training_gate",
                "safety_flags",
                "empty_reason",
            )
            if key in replay_dataset_bundle_body
        }

    replay_dataset_history_summary: dict[str, Any] | None = None
    replay_dataset_history_body = response_bodies.get("replay_dataset_history")
    if isinstance(replay_dataset_history_body, dict):
        replay_dataset_history_summary = {
            key: replay_dataset_history_body.get(key)
            for key in (
                "export_kind",
                "schema_version",
                "training_role",
                "created_at",
                "endpoint",
                "limit",
                "count",
                "source_endpoint",
                "replay_sample_summary",
                "safety_flags",
            )
            if key in replay_dataset_history_body
        }
        replay_dataset_history_summary["latest_history_timestamp"] = _latest_replay_dataset_history_timestamp(
            replay_dataset_history_body
        )

    policy_actuator_summary: dict[str, Any] | None = None
    policy_body = response_bodies.get("policy_actuator")
    if isinstance(policy_body, dict):
        policy_actuator_summary = {
            key: policy_body.get(key)
            for key in (
                "schema_version",
                "action",
                "recommendation",
                "risk",
                "expected_information_gain",
                "expected_goal_progress",
                "expected_cost",
                "uncertainty",
                "advisory",
                "executable",
                "target_episode_id",
                "target_action_id",
                "suggested_endpoint",
            )
            if key in policy_body
        }
        reasons = policy_body.get("reasons")
        if isinstance(reasons, list):
            policy_actuator_summary["reason_codes"] = [
                str(item.get("code"))
                for item in reasons
                if isinstance(item, dict) and item.get("code") is not None
            ]

    feed_summary: dict[str, Any] | None = None
    feed_body = response_bodies.get("feed")
    if isinstance(feed_body, dict) and isinstance(feed_body.get("feed_summary"), dict):
        feed_summary = dict(feed_body["feed_summary"])

    replay_plan_summary: dict[str, Any] | None = None
    replay_body = response_bodies.get("replay_plan")
    if isinstance(replay_body, dict):
        replay_plan_summary = {
            key: replay_body.get(key)
            for key in (
                "schema_version",
                "generated_at",
                "advisory",
                "executable",
                "endpoint",
                "limit",
                "count",
                "priority_rules_version",
                "plan_reason_codes",
                "snapshot_counts",
            )
            if key in replay_body
        }
        candidates = replay_body.get("candidates")
        if isinstance(candidates, list) and candidates:
            top = candidates[0]
            if isinstance(top, dict):
                replay_plan_summary["top_candidate"] = {
                    key: top.get(key)
                    for key in (
                        "candidate_id",
                        "rank",
                        "target_type",
                        "target_id",
                        "operation",
                        "priority_score",
                        "reason_codes",
                        "suggested_consolidation_action",
                        "suggested_endpoint",
                    )
                    if key in top
                }

    replay_sample_summary: dict[str, Any] | None = None
    replay_history_body = response_bodies.get("replay_sample_history")
    replay_sample_summary = _summarize_replay_sample_history(
        replay_history_body,
        fallback=living_loop if isinstance(living_loop, dict) else export_body,
    )
    if replay_sample_summary is None and isinstance(export_body, dict):
        replay_sample_summary = _summarize_replay_sample_history(None, fallback=export_body)

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    result: dict[str, Any] = {
        "benchmark": "hecsn_service_endpoint_latency",
        "schema_version": SERVICE_BENCHMARK_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path)),
        "success": all(bool(item.get("success")) for item in endpoint_timings),
        "total_latency_ms": round(float(total_latency_ms), 3),
        "endpoint_timings": endpoint_timings,
        "endpoints_by_name": {str(item["name"]): item for item in endpoint_timings},
        "living_loop_benchmark_telemetry": living_loop_telemetry,
        "feed_summary": feed_summary,
        "status_runtime_truth_summary": status_runtime_truth_summary,
        "terminus_runtime_truth_summary": terminus_runtime_truth_summary,
        "source_configuration_evidence": {
            "status": status_source_configuration,
            "terminus": terminus_source_configuration,
            "semantics": {
                "benchmark_runtime_configuration": "in_process_service_app_current_manager_state",
                "long_test_difference": (
                    "Long-test calls quick_start_terminus before sampling; service benchmark reports the app state it was given. "
                    "If configured=false here, configure_terminus_sources is an actionable benchmark setup instruction, not a contradiction with a separately configured long-test run."
                ),
            },
        },
        "feedback_telemetry": feedback_telemetry,
        "policy_actuator_summary": policy_actuator_summary,
        "replay_plan_summary": replay_plan_summary,
        "replay_sample_summary": replay_sample_summary,
        "replay_executor_summary": replay_sample_summary,
        "trace_export_summary": export_summary,
        "replay_dataset_summary": replay_dataset_summary,
        "replay_dataset_bundle_summary": replay_dataset_bundle_summary,
        "replay_dataset_candidates_summary": replay_dataset_candidates_summary,
        "replay_dataset_history_summary": replay_dataset_history_summary,
    }
    result["output_path"] = str(Path(output_path))
    _write_json(output_path, result)
    return result


def run_service_benchmark(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    trace_history_limit: int = 32,
    trace_dir: str | Path | None = None,
    web_dist_dir: str | Path | None = None,
    env_root: str | Path | None = None,
    feed_text: str = DEFAULT_FEED_TEXT,
    query_text: str = DEFAULT_QUERY_TEXT,
    top_k_candidates: int = 5,
    top_k_memories: int = 5,
    top_chars: int = 6,
    export_limit: int = 3,
    include_living_loop_telemetry: bool = True,
) -> dict[str, Any]:
    app = create_app(
        checkpoint_path=checkpoint_path,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir,
        web_dist_dir=web_dist_dir,
        env_root=env_root,
    )
    return benchmark_service_app(
        app,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        feed_text=feed_text,
        query_text=query_text,
        top_k_candidates=top_k_candidates,
        top_k_memories=top_k_memories,
        top_chars=top_chars,
        export_limit=export_limit,
        include_living_loop_telemetry=include_living_loop_telemetry,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local HECSN service endpoint latency in-process.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--web-dist-dir", type=Path, default=None)
    parser.add_argument("--env-root", type=Path, default=None)
    parser.add_argument("--feed-text", type=str, default=DEFAULT_FEED_TEXT)
    parser.add_argument("--query-text", type=str, default=DEFAULT_QUERY_TEXT)
    parser.add_argument("--export-limit", type=int, default=3)
    parser.add_argument(
        "--create-synthetic-checkpoint",
        action="store_true",
        help="Create a tiny deterministic checkpoint at --checkpoint when it does not already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.create_synthetic_checkpoint and not args.checkpoint.exists():
        create_tiny_service_benchmark_checkpoint(args.checkpoint)
    result = run_service_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        trace_dir=args.trace_dir,
        web_dist_dir=args.web_dist_dir,
        env_root=args.env_root,
        feed_text=args.feed_text,
        query_text=args.query_text,
        export_limit=args.export_limit,
    )
    print(json.dumps({"output_path": result["output_path"], "success": result["success"]}, sort_keys=True))


if __name__ == "__main__":
    main()
