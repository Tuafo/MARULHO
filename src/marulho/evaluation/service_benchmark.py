from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marulho.config.model_config import MarulhoConfig
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.service.api import create_app
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


SERVICE_BENCHMARK_SCHEMA_VERSION = 1
SERVICE_BENCHMARK_BASELINE_ARTIFACT_KIND = "marulho_service_benchmark_accepted_baseline"
DEFAULT_FEED_TEXT = "Cats chase mice at night. Cats rest indoors during the day. " * 4
DEFAULT_QUERY_TEXT = "cats chase mice"
DEFAULT_LOCAL_SOURCE_TEXT = (
    "Adaptive memory plasticity stabilizes sparse spike routing. "
    "Grounded local observations support prediction error and replay readiness. "
) * 8
SETUP_ENDPOINTS = frozenset({"brain_feed_configured", "brain_tick_configured"})
HOT_PATH_ENDPOINTS = frozenset({"brain_feed", "brain_tick", "brain_generate"})
STATUS_ENDPOINTS = frozenset({"health", "brain_status", "brain_checkpoints", "brain_traces"})
SLOW_PATH_ENDPOINTS = frozenset(
    {
        "brain_replay",
        "brain_grow_prune",
    }
)
HOT_PATH_P95_BUDGET_MS = 1000.0
HOT_PATH_TOTAL_BUDGET_MS = 3000.0
RUNTIME_TRUTH_ORDER = {"failed": 0, "degraded": 1, "partial": 2, "alive": 3}
DEFAULT_HOT_PATH_REGRESSION_TOLERANCE = 0.25


@contextmanager
def _temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_tiny_service_benchmark_checkpoint(path: str | Path) -> Path:
    """Create a tiny deterministic checkpoint for local service smoke benchmarks."""
    cfg = MarulhoConfig(
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
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
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
        title="MARULHO Brain Service Benchmark",
    )


def _load_json_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return payload


def _runtime_truth_verdict_from_report(report: Mapping[str, Any]) -> str:
    for key in ("brain_status_summary", "status_runtime_truth_summary", "terminus_runtime_truth_summary", "runtime_truth"):
        value = report.get(key)
        if isinstance(value, Mapping):
            verdict = str(value.get("verdict", "unknown"))
            return verdict if verdict in RUNTIME_TRUTH_ORDER else "unknown"
    return "unknown"


def _metric_float(report: Mapping[str, Any], group: str, metric: str) -> float | None:
    metabolism = report.get("endpoint_metabolism_summary")
    if not isinstance(metabolism, Mapping):
        return None
    group_payload = metabolism.get(group)
    if not isinstance(group_payload, Mapping):
        return None
    value = group_payload.get(metric)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _runtime_device_from_report(report: Mapping[str, Any], source: str = "status") -> Mapping[str, Any]:
    evidence = report.get("runtime_device_evidence")
    if not isinstance(evidence, Mapping):
        return {}
    payload = evidence.get(source)
    return payload if isinstance(payload, Mapping) else {}


def _endpoint_success_names(report: Mapping[str, Any]) -> set[str]:
    timings = report.get("endpoint_timings")
    if not isinstance(timings, list):
        return set()
    return {
        str(item.get("name"))
        for item in timings
        if isinstance(item, Mapping) and bool(item.get("success")) and item.get("name") is not None
    }


def _hot_path_delta(cpu_report: Mapping[str, Any], cuda_report: Mapping[str, Any]) -> dict[str, Any]:
    cpu_p95 = _metric_float(cpu_report, "hot_path", "latency_ms_p95")
    cuda_p95 = _metric_float(cuda_report, "hot_path", "latency_ms_p95")
    cpu_total = _metric_float(cpu_report, "hot_path", "latency_ms_total")
    cuda_total = _metric_float(cuda_report, "hot_path", "latency_ms_total")

    def ratio(cpu_value: float | None, cuda_value: float | None) -> float | None:
        if cpu_value is None or cuda_value is None or cuda_value <= 0.0:
            return None
        return round(float(cpu_value) / float(cuda_value), 4)

    def delta(cpu_value: float | None, cuda_value: float | None) -> float | None:
        if cpu_value is None or cuda_value is None:
            return None
        return round(float(cuda_value) - float(cpu_value), 3)

    return {
        "cpu_p95_ms": cpu_p95,
        "cuda_p95_ms": cuda_p95,
        "p95_delta_ms_cuda_minus_cpu": delta(cpu_p95, cuda_p95),
        "p95_cpu_over_cuda_ratio": ratio(cpu_p95, cuda_p95),
        "cpu_total_ms": cpu_total,
        "cuda_total_ms": cuda_total,
        "total_delta_ms_cuda_minus_cpu": delta(cpu_total, cuda_total),
        "total_cpu_over_cuda_ratio": ratio(cpu_total, cuda_total),
    }


def _allowed_after_value(before_value: float | None, tolerance: float) -> float | None:
    if before_value is None:
        return None
    return round(float(before_value) * (1.0 + max(0.0, float(tolerance))), 3)


def _report_group_endpoint_names(report: Mapping[str, Any], group: str) -> list[str]:
    metabolism = report.get("endpoint_metabolism_summary")
    if not isinstance(metabolism, Mapping):
        return []
    payload = metabolism.get(group)
    if not isinstance(payload, Mapping):
        return []
    names = payload.get("endpoint_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def _check_report_success(report: Mapping[str, Any]) -> bool:
    if "success" in report:
        return bool(report.get("success"))
    timings = report.get("endpoint_timings")
    return isinstance(timings, list) and all(
        isinstance(item, Mapping) and bool(item.get("success")) for item in timings
    )


def compare_service_benchmark_reports(
    *,
    before_report: Mapping[str, Any],
    after_report: Mapping[str, Any],
    hot_path_regression_tolerance: float = DEFAULT_HOT_PATH_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    before_verdict = _runtime_truth_verdict_from_report(before_report)
    after_verdict = _runtime_truth_verdict_from_report(after_report)
    runtime_truth_regressed = (
        before_verdict in RUNTIME_TRUTH_ORDER
        and after_verdict in RUNTIME_TRUTH_ORDER
        and RUNTIME_TRUTH_ORDER[after_verdict] < RUNTIME_TRUTH_ORDER[before_verdict]
    )
    before_hot_p95 = _metric_float(before_report, "hot_path", "latency_ms_p95")
    after_hot_p95 = _metric_float(after_report, "hot_path", "latency_ms_p95")
    before_hot_total = _metric_float(before_report, "hot_path", "latency_ms_total")
    after_hot_total = _metric_float(after_report, "hot_path", "latency_ms_total")
    allowed_hot_p95 = _allowed_after_value(before_hot_p95, hot_path_regression_tolerance)
    allowed_hot_total = _allowed_after_value(before_hot_total, hot_path_regression_tolerance)
    configured = after_report.get("configured_source_summary")
    configured_source = configured if isinstance(configured, Mapping) else {}
    source_config = after_report.get("source_configuration_evidence")
    source_status = (
        source_config.get("status")
        if isinstance(source_config, Mapping) and isinstance(source_config.get("status"), Mapping)
        else {}
    )
    hot_names = set(_report_group_endpoint_names(after_report, "hot_path"))
    setup_names = set(_report_group_endpoint_names(after_report, "setup"))
    slow_names = set(_report_group_endpoint_names(after_report, "slow_path"))
    setup_leaked = bool(hot_names & SETUP_ENDPOINTS)
    slow_leaked = bool(hot_names & SLOW_PATH_ENDPOINTS)
    checks = {
        "before_success": _check_report_success(before_report),
        "after_success": _check_report_success(after_report),
        "runtime_truth_no_regression": not runtime_truth_regressed,
        "configured_source_alive": after_verdict == "alive"
        and bool(configured_source.get("configured"))
        and int(configured_source.get("tick_tokens_processed", 0) or 0) > 0
        and bool(source_status.get("configured")),
        "hot_path_p95_within_absolute_budget": isinstance(after_hot_p95, float)
        and after_hot_p95 <= HOT_PATH_P95_BUDGET_MS,
        "hot_path_total_within_absolute_budget": isinstance(after_hot_total, float)
        and after_hot_total <= HOT_PATH_TOTAL_BUDGET_MS,
        "hot_path_p95_no_relative_regression": (
            isinstance(after_hot_p95, float)
            and (allowed_hot_p95 is None or after_hot_p95 <= allowed_hot_p95)
        ),
        "hot_path_total_no_relative_regression": (
            isinstance(after_hot_total, float)
            and (allowed_hot_total is None or after_hot_total <= allowed_hot_total)
        ),
        "setup_not_in_hot_path": not setup_leaked,
        "slow_path_not_in_hot_path": not slow_leaked,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "marulho_service_benchmark_regression_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "runtime_truth": {
            "before": before_verdict,
            "after": after_verdict,
            "regressed": runtime_truth_regressed,
        },
        "hot_path": {
            "regression_tolerance": float(hot_path_regression_tolerance),
            "before_p95_ms": before_hot_p95,
            "after_p95_ms": after_hot_p95,
            "allowed_after_p95_ms": allowed_hot_p95,
            "absolute_p95_budget_ms": HOT_PATH_P95_BUDGET_MS,
            "before_total_ms": before_hot_total,
            "after_total_ms": after_hot_total,
            "allowed_after_total_ms": allowed_hot_total,
            "absolute_total_budget_ms": HOT_PATH_TOTAL_BUDGET_MS,
        },
        "endpoint_grouping": {
            "hot_path": sorted(hot_names),
            "setup": sorted(setup_names),
            "slow_path": sorted(slow_names),
            "setup_leaked_into_hot_path": setup_leaked,
            "slow_path_leaked_into_hot_path": slow_leaked,
        },
        "configured_source": {
            "source_name": configured_source.get("source_name"),
            "configured": bool(configured_source.get("configured")),
            "tick_tokens_processed": int(configured_source.get("tick_tokens_processed", 0) or 0),
            "source_count": int(source_status.get("source_count", 0) or 0),
            "source_names": list(source_status.get("source_names", []))
            if isinstance(source_status.get("source_names"), list)
            else [],
        },
        "claim_boundary": "regression_gate_only_no_runtime_mutation_no_cuda_speedup_claim",
    }


def compare_service_benchmark_report_files(
    *,
    before_path: str | Path,
    after_path: str | Path,
    output_path: str | Path | None = None,
    hot_path_regression_tolerance: float = DEFAULT_HOT_PATH_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    comparison = compare_service_benchmark_reports(
        before_report=_load_json_report(before_path),
        after_report=_load_json_report(after_path),
        hot_path_regression_tolerance=hot_path_regression_tolerance,
    )
    if output_path is not None:
        _write_json(output_path, comparison)
    return comparison


def _baseline_acceptance_material(
    *,
    baseline_id: str,
    label: str,
    accepted_by: str,
    note: str,
    accepted_at: str,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "baseline_id": baseline_id,
        "label": label,
        "accepted_by": accepted_by,
        "note": note,
        "accepted_at": accepted_at,
        "source_report_sha256_canonical_json": source_report.get("sha256_canonical_json", ""),
        "source_report_generated_at": source_report.get("generated_at", ""),
        "runtime_truth_verdict": source_report.get("runtime_truth_verdict", ""),
        "hot_path_p95_ms": source_report.get("hot_path_p95_ms"),
        "hot_path_total_ms": source_report.get("hot_path_total_ms"),
    }


def _validate_baseline_acceptance_hash(baseline: Mapping[str, Any]) -> None:
    operator_review = baseline.get("operator_review")
    if not isinstance(operator_review, Mapping):
        return
    expected_hash = str(operator_review.get("acceptance_hash", "") or "")
    if not expected_hash:
        return
    material = operator_review.get("acceptance_material")
    if not isinstance(material, Mapping):
        raise ValueError("Baseline operator acceptance hash is present but acceptance material is missing.")
    actual_hash = _sha256_json(material)
    if actual_hash != expected_hash:
        raise ValueError("Baseline operator acceptance hash does not match acceptance material.")


def create_service_benchmark_accepted_baseline(
    *,
    report_path: str | Path,
    output_path: str | Path,
    accepted_by: str,
    label: str = "",
    note: str = "",
) -> dict[str, Any]:
    benchmark_report = _load_json_report(report_path)
    hot_path_p95 = _metric_float(benchmark_report, "hot_path", "latency_ms_p95")
    hot_path_total = _metric_float(benchmark_report, "hot_path", "latency_ms_total")
    runtime_truth = _runtime_truth_verdict_from_report(benchmark_report)
    checks = {
        "accepted_by_present": bool(str(accepted_by).strip()),
        "benchmark_success": _check_report_success(benchmark_report),
        "runtime_truth_known": runtime_truth in RUNTIME_TRUTH_ORDER,
        "hot_path_p95_available": isinstance(hot_path_p95, float),
        "hot_path_total_available": isinstance(hot_path_total, float),
    }
    if not all(checks.values()):
        failed = ", ".join(sorted(name for name, passed in checks.items() if not passed))
        raise ValueError(f"Cannot accept service benchmark baseline; failed checks: {failed}")
    report_hash = _sha256_json(benchmark_report)
    generated_at = datetime.now(timezone.utc).isoformat()
    accepted_at = datetime.now(timezone.utc).isoformat()
    baseline_id = f"service-benchmark-baseline:{report_hash[:16]}"
    baseline_label = str(label).strip()
    reviewer = str(accepted_by).strip()
    review_note = str(note).strip()
    source_report = {
        "path": str(Path(report_path)),
        "sha256_canonical_json": report_hash,
        "generated_at": benchmark_report.get("generated_at", ""),
        "runtime_truth_verdict": runtime_truth,
        "hot_path_p95_ms": hot_path_p95,
        "hot_path_total_ms": hot_path_total,
    }
    acceptance_material = _baseline_acceptance_material(
        baseline_id=baseline_id,
        label=baseline_label,
        accepted_by=reviewer,
        note=review_note,
        accepted_at=accepted_at,
        source_report=source_report,
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": SERVICE_BENCHMARK_BASELINE_ARTIFACT_KIND,
        "generated_at": generated_at,
        "status": "accepted",
        "baseline_id": baseline_id,
        "label": baseline_label,
        "operator_review": {
            "accepted_by": reviewer,
            "note": review_note,
            "accepted_at": accepted_at,
            "acceptance_material": acceptance_material,
            "acceptance_hash": _sha256_json(acceptance_material),
            "acceptance_hash_algorithm": "sha256_canonical_json",
        },
        "checks": checks,
        "source_report": source_report,
        "baseline_report_snapshot": benchmark_report,
        "claim_boundary": "accepted_baseline_manifest_only_no_runtime_mutation_no_cuda_speedup_claim",
    }
    _write_json(output_path, manifest)
    return manifest


def compare_service_benchmark_against_accepted_baseline(
    *,
    baseline_path: str | Path,
    after_path: str | Path,
    output_path: str | Path | None = None,
    hot_path_regression_tolerance: float = DEFAULT_HOT_PATH_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    baseline = _load_json_report(baseline_path)
    if baseline.get("artifact_kind") != SERVICE_BENCHMARK_BASELINE_ARTIFACT_KIND:
        raise ValueError("Baseline manifest is not a service benchmark accepted baseline.")
    _validate_baseline_acceptance_hash(baseline)
    baseline_report = baseline.get("baseline_report_snapshot")
    if not isinstance(baseline_report, Mapping):
        raise ValueError("Baseline manifest is missing baseline_report_snapshot.")
    source_report = baseline.get("source_report")
    expected_hash = (
        str(source_report.get("sha256_canonical_json", ""))
        if isinstance(source_report, Mapping)
        else ""
    )
    actual_hash = _sha256_json(baseline_report)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("Baseline report snapshot hash does not match baseline source hash.")
    after_report = _load_json_report(after_path)
    comparison = compare_service_benchmark_reports(
        before_report=baseline_report,
        after_report=after_report,
        hot_path_regression_tolerance=hot_path_regression_tolerance,
    )
    comparison["accepted_baseline"] = {
        "baseline_path": str(Path(baseline_path)),
        "baseline_id": baseline.get("baseline_id", ""),
        "label": baseline.get("label", ""),
        "accepted_by": baseline.get("operator_review", {}).get("accepted_by", "")
        if isinstance(baseline.get("operator_review"), Mapping)
        else "",
        "baseline_report_hash": actual_hash,
        "after_report_hash": _sha256_json(after_report),
        "claim_boundary": "accepted_baseline_used_for_report_only_regression_gate",
    }
    if output_path is not None:
        _write_json(output_path, comparison)
    return comparison


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 3)


def _latency_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item.get("latency_ms", 0.0) or 0.0) for item in records]
    response_sizes = [int(item.get("response_size_bytes", 0) or 0) for item in records]
    failures = [str(item.get("name", "")) for item in records if not bool(item.get("success"))]
    return {
        "count": int(len(records)),
        "success": not failures,
        "failed_endpoints": failures,
        "latency_ms_total": round(float(sum(latencies)), 3),
        "latency_ms_mean": round(float(sum(latencies) / len(latencies)), 3) if latencies else None,
        "latency_ms_max": round(float(max(latencies)), 3) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 50.0),
        "latency_ms_p95": _percentile(latencies, 95.0),
        "response_size_bytes_total": int(sum(response_sizes)),
        "endpoint_names": [str(item.get("name", "")) for item in records],
    }


def _endpoint_metabolism_summary(endpoint_timings: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "setup": [item for item in endpoint_timings if str(item.get("name")) in SETUP_ENDPOINTS],
        "hot_path": [item for item in endpoint_timings if str(item.get("name")) in HOT_PATH_ENDPOINTS],
        "status": [item for item in endpoint_timings if str(item.get("name")) in STATUS_ENDPOINTS],
        "slow_path": [item for item in endpoint_timings if str(item.get("name")) in SLOW_PATH_ENDPOINTS],
    }
    grouped_names = set().union(*(set(str(item.get("name")) for item in records) for records in groups.values()))
    uncategorized = [item for item in endpoint_timings if str(item.get("name")) not in grouped_names]
    summary = {name: _latency_group_summary(records) for name, records in groups.items()}
    hot_path = summary["hot_path"]
    hot_path_p95 = hot_path.get("latency_ms_p95")
    hot_path_total = hot_path.get("latency_ms_total")
    within_budget = (
        bool(hot_path.get("success"))
        and isinstance(hot_path_p95, (int, float))
        and isinstance(hot_path_total, (int, float))
        and float(hot_path_p95) <= HOT_PATH_P95_BUDGET_MS
        and float(hot_path_total) <= HOT_PATH_TOTAL_BUDGET_MS
    )
    summary["hot_path_budget"] = {
        "p95_budget_ms": HOT_PATH_P95_BUDGET_MS,
        "total_budget_ms": HOT_PATH_TOTAL_BUDGET_MS,
        "within_budget": bool(within_budget),
        "hot_path_protection_role": "benchmark_evidence_only_not_runtime_work",
    }
    summary["uncategorized"] = _latency_group_summary(uncategorized)
    summary["semantics"] = {
        "setup_endpoints": sorted(SETUP_ENDPOINTS),
        "hot_path_endpoints": sorted(HOT_PATH_ENDPOINTS),
        "status_endpoints": sorted(STATUS_ENDPOINTS),
        "slow_path_endpoints": sorted(SLOW_PATH_ENDPOINTS),
        "setup_note": "Configuration and manual tick are benchmark setup work, not always-on runtime hot-path work.",
        "slow_path_note": "Runtime trace export is an explicit tooling/evaluation surface, not always-on runtime work.",
    }
    return summary


def _summarize_runtime_truth(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    if body.get("surface") == "marulho_brain_runtime.v1":
        return {
            "schema_version": 1,
            "verdict": "alive",
            "recommended_action": "continue_brain_loop",
            "surface": body.get("surface"),
            "evidence": {
                "token_count": int(body.get("token_count", 0) or 0),
                "queued_tokens": int(body.get("queued_tokens", 0) or 0),
                "executor": body.get("executor"),
                "route_vote_mode": body.get("route_vote_mode"),
                "cuda_available": bool(body.get("cuda_available", False)),
                "readout": body.get("readout"),
                "loop": body.get("loop"),
            },
        }
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
    if body.get("surface") == "marulho_brain_runtime.v1":
        source_buffer = body.get("source_buffer") if isinstance(body.get("source_buffer"), dict) else {}
        return {
            "configured": True,
            "source_count": 1 if int(source_buffer.get("queued_tokens", body.get("queued_tokens", 0)) or 0) > 0 else 0,
            "source_names": [],
            "source_types": ["brain_source_buffer"],
            "queued_tokens": int(body.get("queued_tokens", 0) or 0),
            "tick_tokens": None,
            "repeat_sources": False,
            "ingestion": {"surface": "marulho_brain_source_buffer.v1"},
        }
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


def _summarize_runtime_device_evidence(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    if body.get("surface") == "marulho_brain_runtime.v1":
        device = str(body.get("device") or "")
        cuda_available = bool(body.get("cuda_available", False))
        observed_cuda_execution = device.startswith("cuda")
        return {
            "summary_role": "observed_brain_device_evidence_not_acceleration_claim",
            "requested_device": None,
            "env_device": None,
            "resolved_device": device,
            "cuda_available": cuda_available,
            "cuda_selected": observed_cuda_execution,
            "cuda_device_count": 1 if cuda_available else 0,
            "tensor_device": device,
            "routing_search_device": None,
            "routing_backend_cuda_capable": None,
            "encoder": None,
            "encoder_device": None,
            "subcortex_device_sections": ["trainer"],
            "observed_cuda_execution": observed_cuda_execution,
            "cuda_fallback_reason": None if observed_cuda_execution else ("cuda_available_but_not_selected_by_brain" if cuda_available else "cuda_not_available"),
            "unit_tests_default_cpu": not observed_cuda_execution,
        }
    runtime_scope = body.get("runtime_scope")
    if not isinstance(runtime_scope, dict):
        return None
    cuda_runtime = runtime_scope.get("cuda_first_runtime")
    if not isinstance(cuda_runtime, dict):
        return None
    device_config = runtime_scope.get("device")
    if not isinstance(device_config, dict):
        device_config = {}
    encoder_report = cuda_runtime.get("encoder_device_report")
    if not isinstance(encoder_report, dict):
        encoder_report = {}
    subcortex_devices = cuda_runtime.get("subcortex_tensor_devices")
    if not isinstance(subcortex_devices, dict):
        subcortex_devices = {}
    tensor_device = str(cuda_runtime.get("tensor_device") or "")
    routing_search_device = str(cuda_runtime.get("routing_search_device") or "")
    encoder_device = str(encoder_report.get("device") or "")
    resolved_device = str(device_config.get("resolved_device") or tensor_device)
    cuda_available = bool(device_config.get("cuda_available", False))
    cuda_selected = bool(device_config.get("cuda_selected", False))
    tensor_device_is_cuda = tensor_device.startswith("cuda")
    routing_device_is_cuda = routing_search_device.startswith("cuda")
    encoder_device_is_cuda = encoder_device.startswith("cuda")
    observed_cuda_execution = bool(tensor_device_is_cuda or routing_device_is_cuda or encoder_device_is_cuda)
    cuda_fallback_reason = None
    if not observed_cuda_execution:
        if cuda_available:
            cuda_fallback_reason = "cuda_available_but_not_selected_by_runtime_scope"
        else:
            cuda_fallback_reason = "cuda_not_available"
    return {
        "summary_role": "observed_runtime_device_evidence_not_acceleration_claim",
        "requested_device": device_config.get("requested_device"),
        "env_device": device_config.get("env_device"),
        "resolved_device": resolved_device,
        "cuda_available": cuda_available,
        "cuda_selected": cuda_selected,
        "cuda_device_count": int(device_config.get("cuda_device_count", 0) or 0),
        "tensor_device": tensor_device,
        "routing_search_device": routing_search_device,
        "routing_backend_cuda_capable": bool(cuda_runtime.get("routing_backend_cuda_capable", False)),
        "encoder": encoder_report.get("encoder"),
        "encoder_device": encoder_device,
        "subcortex_device_sections": sorted(str(key) for key in subcortex_devices.keys()),
        "observed_cuda_execution": observed_cuda_execution,
        "cuda_fallback_reason": cuda_fallback_reason,
        "unit_tests_default_cpu": bool(cuda_runtime.get("unit_tests_default_cpu", False)),
    }


def benchmark_service_app(
    app: FastAPI,
    *,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    configured_source_path: str | Path | None = None,
    configured_source_name: str = "benchmark_local_source",
    configured_source_tick_steps: int = 0,
    configured_source_tick_tokens: int = 128,
    configured_source_queue_target_tokens: int | None = None,
    configured_source_prewarm_on_startup: bool = True,
    configured_source_prewarm_wait_seconds: float = 5.0,
    feed_text: str = DEFAULT_FEED_TEXT,
    query_text: str = DEFAULT_QUERY_TEXT,
    top_k_candidates: int = 5,
    top_k_memories: int = 5,
    top_chars: int = 6,
    export_limit: int = 3,
    profile_trainer_stages: bool = False,
) -> dict[str, Any]:
    """Measure the local service endpoints in-process and write JSON results."""
    started = time.perf_counter()
    endpoint_timings: list[dict[str, Any]] = []
    response_bodies: dict[str, Any] = {}
    configured_source_warmup_evidence: dict[str, Any] | None = None
    manager = getattr(getattr(app, "state", None), "marulho_manager", None)
    brain = getattr(manager, "brain", None)
    trainer = getattr(brain, "trainer", None) or getattr(manager, "_trainer", None)
    captured_trainer_stage_profile: dict[str, Any] | None = None
    profile_configured_tick_only = bool(
        profile_trainer_stages
        and configured_source_path is not None
        and int(configured_source_tick_steps) > 0
    )
    if bool(profile_trainer_stages) and not profile_configured_tick_only and trainer is not None:
        enable_profile = getattr(trainer, "enable_train_step_profile", None)
        if callable(enable_profile):
            enable_profile(reset=True)

    with TestClient(app) as client:
        if configured_source_path is not None:
            tick_tokens = max(1, int(configured_source_tick_tokens))
            source_text = Path(configured_source_path).read_text(encoding="utf-8")
            feed_record, feed_body = _measure_endpoint(
                client,
                name="brain_feed_configured",
                method="POST",
                path="/brain/feed",
                json_body={
                    "text": source_text,
                    "source": str(configured_source_name),
                    "learn": False,
                },
            )
            endpoint_timings.append(feed_record)
            response_bodies["brain_feed_configured"] = feed_body
            configured_source_warmup_evidence = {
                "enabled": bool(configured_source_prewarm_on_startup),
                "mode": "brain_source_buffer_feed_no_terminus_prewarm",
                "not_hot_path": True,
                "wait_budget_seconds": float(configured_source_prewarm_wait_seconds),
                "queue_target_tokens": (
                    int(configured_source_queue_target_tokens)
                    if configured_source_queue_target_tokens is not None
                    else tick_tokens
                ),
            }
            if int(configured_source_tick_steps) > 0:
                if profile_configured_tick_only and trainer is not None:
                    enable_profile = getattr(trainer, "enable_train_step_profile", None)
                    if callable(enable_profile):
                        enable_profile(reset=True)
                tick_record, tick_body = _measure_endpoint(
                    client,
                    name="brain_tick_configured",
                    method="POST",
                    path="/brain/tick",
                    json_body={
                        "tokens": tick_tokens * max(1, int(configured_source_tick_steps)),
                        "source": str(configured_source_name),
                    },
                )
                if profile_configured_tick_only and trainer is not None:
                    report_profile = getattr(trainer, "train_step_profile_report", None)
                    if callable(report_profile):
                        captured_trainer_stage_profile = dict(report_profile())
                    disable_profile = getattr(trainer, "disable_train_step_profile", None)
                    if callable(disable_profile):
                        disable_profile()
                endpoint_timings.append(tick_record)
                response_bodies["brain_tick_configured"] = tick_body

        requests: tuple[dict[str, Any], ...] = (
            {"name": "health", "method": "GET", "path": "/health"},
            {"name": "brain_status", "method": "GET", "path": "/brain/status"},
            {"name": "brain_checkpoints", "method": "GET", "path": "/brain/checkpoints"},
            {
                "name": "brain_feed",
                "method": "POST",
                "path": "/brain/feed",
                "json_body": {"text": feed_text, "source": "benchmark"},
            },
            {
                "name": "brain_tick",
                "method": "POST",
                "path": "/brain/tick",
                "json_body": {"tokens": max(1, int(top_k_candidates) + int(top_k_memories)), "source": "benchmark"},
            },
            {
                "name": "brain_generate",
                "method": "POST",
                "path": "/brain/generate",
                "json_body": {
                    "prompt": query_text,
                    "max_tokens": max(1, int(top_chars) * 2),
                },
            },
            {"name": "brain_traces", "method": "GET", "path": "/brain/traces", "params": {"limit": int(export_limit)}},
            {
                "name": "brain_replay",
                "method": "POST",
                "path": "/brain/replay",
                "json_body": {"window": "micro", "cycles": 1},
            },
            {
                "name": "brain_grow_prune",
                "method": "POST",
                "path": "/brain/grow-prune",
                "json_body": {"budget": "small"},
            },
        )
        for request in requests:
            record, body = _measure_endpoint(client, **request)
            endpoint_timings.append(record)
            response_bodies[str(request["name"])] = body

    feedback_telemetry: dict[str, Any] | None = None
    brain_status_summary = _summarize_runtime_truth(response_bodies.get("brain_status"))
    status_runtime_truth_summary = brain_status_summary
    terminus_runtime_truth_summary = None
    brain_source_configuration = _summarize_source_configuration(response_bodies.get("brain_status"))
    status_source_configuration = brain_source_configuration
    terminus_source_configuration = None
    brain_device_evidence = _summarize_runtime_device_evidence(response_bodies.get("brain_status"))
    status_device_evidence = brain_device_evidence
    terminus_device_evidence = None

    export_summary: dict[str, Any] | None = None
    traces_body = response_bodies.get("brain_traces")
    if isinstance(traces_body, dict):
        traces = traces_body.get("traces")
        export_summary = {
            "surface": traces_body.get("surface"),
            "limit": int(export_limit),
            "count": len(traces) if isinstance(traces, list) else None,
            "endpoint": "/brain/traces",
        }

    feed_summary: dict[str, Any] | None = None
    feed_body = response_bodies.get("brain_feed")
    if isinstance(feed_body, dict) and isinstance(feed_body.get("feed_summary"), dict):
        feed_summary = dict(feed_body["feed_summary"])
    elif isinstance(feed_body, dict):
        feed_summary = {
            "surface": feed_body.get("surface"),
            "tokens_processed": int(feed_body.get("accepted_tokens", 0) or 0),
            "queued_tokens": int(feed_body.get("queued_tokens", 0) or 0),
            "source": feed_body.get("source"),
        }

    configured_source_summary = None
    if configured_source_path is not None:
        configure_body = response_bodies.get("brain_feed_configured")
        tick_body = response_bodies.get("brain_tick_configured")
        tick_trace = tick_body.get("trace") if isinstance(tick_body, dict) and isinstance(tick_body.get("trace"), dict) else {}
        source_cache_summary = {
            "cache_write_count": 0,
            "cache_schedule_count": 0,
            "cache_skip_count": 0,
            "cache_failure_count": 0,
            "cache_pending": False,
            "last_cache_update_mode": "not_applicable_brain_source_buffer",
        }
        configured_source_summary = {
            "enabled": True,
            "source_name": str(configured_source_name),
            "source_path": str(Path(configured_source_path)),
            "tick_steps": int(configured_source_tick_steps),
            "configure_success": bool(response_bodies.get("brain_feed_configured") is not None),
            "tick_success": bool(response_bodies.get("brain_tick_configured") is not None) if int(configured_source_tick_steps) > 0 else None,
            "configured": True,
            "source_count": 1,
            "accepted_tokens": int(configure_body.get("accepted_tokens", 0) or 0) if isinstance(configure_body, dict) else 0,
            "tick_tokens_processed": int(tick_body.get("trained_tokens", 0) or 0) if isinstance(tick_body, dict) else 0,
            "background_tokens_processed": int(tick_body.get("trained_tokens", 0) or 0) if isinstance(tick_body, dict) else 0,
            "last_tick_duration_ms": tick_trace.get("elapsed_ms"),
            "stage_timings_ms": {},
            "concept_observation": {"mode": "not_applicable_brain_trace"},
            "source_cache": source_cache_summary,
            "warmup": dict(configured_source_warmup_evidence or {"enabled": False}),
            "not_hot_path": True,
        }

    trainer_stage_profile: dict[str, Any] | None = None
    if bool(profile_trainer_stages):
        if captured_trainer_stage_profile is not None:
            trainer_stage_profile = captured_trainer_stage_profile
            trainer_stage_profile["scope"] = (
                "configured_source_tick"
                if profile_configured_tick_only
                else "benchmark_app"
            )
        elif trainer is not None and callable(getattr(trainer, "train_step_profile_report", None)):
            trainer_stage_profile = dict(trainer.train_step_profile_report())
            trainer_stage_profile["scope"] = "benchmark_app"
            disable_profile = getattr(trainer, "disable_train_step_profile", None)
            if callable(disable_profile):
                disable_profile()
        else:
            trainer_stage_profile = {
                "enabled": False,
                "count": 0,
                "unavailable_reason": "app_state_marulho_manager_trainer_missing",
            }

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    endpoint_metabolism_summary = _endpoint_metabolism_summary(endpoint_timings)
    result: dict[str, Any] = {
        "benchmark": "marulho_service_endpoint_latency",
        "schema_version": SERVICE_BENCHMARK_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path)),
        "success": all(bool(item.get("success")) for item in endpoint_timings),
        "total_latency_ms": round(float(total_latency_ms), 3),
        "endpoint_timings": endpoint_timings,
        "endpoints_by_name": {str(item["name"]): item for item in endpoint_timings},
        "endpoint_metabolism_summary": endpoint_metabolism_summary,
        "trainer_stage_profile": trainer_stage_profile,
        "configured_source_summary": configured_source_summary,
        "feed_summary": feed_summary,
        "brain_status_summary": brain_status_summary,
        "status_runtime_truth_summary": status_runtime_truth_summary,
        "terminus_runtime_truth_summary": terminus_runtime_truth_summary,
        "source_configuration_evidence": {
            "brain": brain_source_configuration,
            "status": status_source_configuration,
            "terminus": terminus_source_configuration,
            "semantics": {
                "benchmark_runtime_configuration": "in_process_brain_service_app_current_state",
                "long_test_difference": (
                    "The brain service benchmark feeds local source text directly into MarulhoBrain; old quick_start_terminus setup is retired."
                ),
            },
        },
        "runtime_device_evidence": {
            "brain": brain_device_evidence,
            "status": status_device_evidence,
            "terminus": terminus_device_evidence,
            "semantics": {
                "claim_boundary": "observed_device_placement_only_not_cuda_speedup",
                "cuda_claim_requires": "observed_cuda_execution_true_plus_cpu_cuda_parity_or_benchmark_delta",
            },
        },
        "feedback_telemetry": feedback_telemetry,
        "trace_export_summary": export_summary,
    }
    result["output_path"] = str(Path(output_path))
    _write_json(output_path, result)
    return result


def run_service_benchmark(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    configure_local_source: bool = False,
    local_source_text: str = DEFAULT_LOCAL_SOURCE_TEXT,
    local_source_tick_steps: int = 0,
    local_source_tick_tokens: int = 128,
    local_source_queue_target_tokens: int | None = None,
    local_source_prewarm_on_startup: bool = True,
    local_source_prewarm_wait_seconds: float = 5.0,
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
    profile_trainer_stages: bool = False,
) -> dict[str, Any]:
    configured_source_path = None
    if configure_local_source:
        source_root = Path(output_path).parent
        source_root.mkdir(parents=True, exist_ok=True)
        configured_source_path = source_root / "benchmark-local-source.txt"
        source_text = str(local_source_text)
        if (
            not configured_source_path.exists()
            or configured_source_path.read_text(encoding="utf-8") != source_text
        ):
            configured_source_path.write_text(source_text, encoding="utf-8")
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
        configured_source_path=configured_source_path,
        configured_source_tick_steps=local_source_tick_steps,
        configured_source_tick_tokens=local_source_tick_tokens,
        configured_source_queue_target_tokens=local_source_queue_target_tokens,
        configured_source_prewarm_on_startup=local_source_prewarm_on_startup,
        configured_source_prewarm_wait_seconds=local_source_prewarm_wait_seconds,
        feed_text=feed_text,
        query_text=query_text,
        top_k_candidates=top_k_candidates,
        top_k_memories=top_k_memories,
        top_chars=top_chars,
        export_limit=export_limit,
        profile_trainer_stages=profile_trainer_stages,
    )


def run_service_benchmark_against_accepted_baseline(
    *,
    checkpoint_path: str | Path,
    baseline_path: str | Path,
    output_dir: str | Path,
    configure_local_source: bool = True,
    local_source_text: str = DEFAULT_LOCAL_SOURCE_TEXT,
    local_source_tick_steps: int = 1,
    local_source_tick_tokens: int = 128,
    local_source_queue_target_tokens: int | None = None,
    local_source_prewarm_on_startup: bool = True,
    local_source_prewarm_wait_seconds: float = 5.0,
    trace_history_limit: int = 32,
    trace_dir: str | Path | None = None,
    web_dist_dir: str | Path | None = None,
    env_root: str | Path | None = None,
    feed_text: str = DEFAULT_FEED_TEXT,
    query_text: str = DEFAULT_QUERY_TEXT,
    export_limit: int = 3,
    hot_path_regression_tolerance: float = DEFAULT_HOT_PATH_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = bundle_dir / "fresh-benchmark.json"
    comparison_path = bundle_dir / "comparison.json"
    summary_path = bundle_dir / "bundle-summary.json"
    benchmark = run_service_benchmark(
        checkpoint_path=checkpoint_path,
        output_path=benchmark_path,
        configure_local_source=configure_local_source,
        local_source_text=local_source_text,
        local_source_tick_steps=local_source_tick_steps,
        local_source_tick_tokens=local_source_tick_tokens,
        local_source_queue_target_tokens=local_source_queue_target_tokens,
        local_source_prewarm_on_startup=local_source_prewarm_on_startup,
        local_source_prewarm_wait_seconds=local_source_prewarm_wait_seconds,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir if trace_dir is not None else bundle_dir / "traces",
        web_dist_dir=web_dist_dir,
        env_root=env_root if env_root is not None else bundle_dir,
        feed_text=feed_text,
        query_text=query_text,
        export_limit=export_limit,
    )
    comparison = compare_service_benchmark_against_accepted_baseline(
        baseline_path=baseline_path,
        after_path=benchmark_path,
        output_path=comparison_path,
        hot_path_regression_tolerance=hot_path_regression_tolerance,
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": "marulho_service_benchmark_baseline_run_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": comparison.get("status", "unknown"),
        "success": bool(benchmark.get("success")) and comparison.get("status") == "passed",
        "paths": {
            "bundle_dir": str(bundle_dir),
            "benchmark": str(benchmark_path),
            "comparison": str(comparison_path),
            "baseline": str(Path(baseline_path)),
        },
        "accepted_baseline": comparison.get("accepted_baseline", {}),
        "runtime_truth": comparison.get("runtime_truth", {}),
        "hot_path": comparison.get("hot_path", {}),
        "configured_source": comparison.get("configured_source", {}),
        "checks": comparison.get("checks", {}),
        "claim_boundary": "fresh_benchmark_plus_baseline_compare_slow_path_no_runtime_mutation_no_cuda_speedup_claim",
    }
    _write_json(summary_path, summary)
    return summary


def compare_service_benchmark_devices(
    *,
    cpu_report: Mapping[str, Any],
    cuda_report: Mapping[str, Any],
    cpu_report_path: str | Path | None = None,
    cuda_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    cpu_device = _runtime_device_from_report(cpu_report)
    cuda_device = _runtime_device_from_report(cuda_report)
    cpu_success_names = _endpoint_success_names(cpu_report)
    cuda_success_names = _endpoint_success_names(cuda_report)
    shared_success_names = sorted(cpu_success_names & cuda_success_names)
    cpu_only_success_names = sorted(cpu_success_names - cuda_success_names)
    cuda_only_success_names = sorted(cuda_success_names - cpu_success_names)
    hot_delta = _hot_path_delta(cpu_report, cuda_report)
    cpu_metabolism = cpu_report.get("endpoint_metabolism_summary")
    cuda_metabolism = cuda_report.get("endpoint_metabolism_summary")
    cpu_budget = (
        cpu_metabolism.get("hot_path_budget")
        if isinstance(cpu_metabolism, Mapping) and isinstance(cpu_metabolism.get("hot_path_budget"), Mapping)
        else {}
    )
    cuda_budget = (
        cuda_metabolism.get("hot_path_budget")
        if isinstance(cuda_metabolism, Mapping) and isinstance(cuda_metabolism.get("hot_path_budget"), Mapping)
        else {}
    )
    checks = {
        "cpu_report_success": _check_report_success(cpu_report),
        "cuda_report_success": _check_report_success(cuda_report),
        "cpu_runtime_truth_alive": _runtime_truth_verdict_from_report(cpu_report) == "alive",
        "cuda_runtime_truth_alive": _runtime_truth_verdict_from_report(cuda_report) == "alive",
        "cpu_observed_not_cuda": not bool(cpu_device.get("observed_cuda_execution", False)),
        "cuda_observed_execution": bool(cuda_device.get("observed_cuda_execution", False)),
        "same_successful_endpoint_names": not cpu_only_success_names and not cuda_only_success_names,
        "hot_path_metrics_available": all(
            isinstance(hot_delta.get(key), (int, float))
            for key in ("cpu_p95_ms", "cuda_p95_ms", "cpu_total_ms", "cuda_total_ms")
        ),
        "cpu_hot_path_within_budget": bool(cpu_budget.get("within_budget", False)),
        "cuda_hot_path_within_budget": bool(cuda_budget.get("within_budget", False)),
    }
    summary = {
        "schema_version": 1,
        "artifact_kind": "marulho_service_benchmark_device_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "success": all(checks.values()),
        "checks": checks,
        "paths": {
            "cpu_report": None if cpu_report_path is None else str(Path(cpu_report_path)),
            "cuda_report": None if cuda_report_path is None else str(Path(cuda_report_path)),
        },
        "runtime_truth": {
            "cpu": _runtime_truth_verdict_from_report(cpu_report),
            "cuda": _runtime_truth_verdict_from_report(cuda_report),
        },
        "runtime_device_evidence": {
            "cpu": dict(cpu_device),
            "cuda": dict(cuda_device),
        },
        "endpoint_success_parity": {
            "shared_success_names": shared_success_names,
            "cpu_only_success_names": cpu_only_success_names,
            "cuda_only_success_names": cuda_only_success_names,
            "parity_scope": "endpoint_success_names_only_not_semantic_output_equivalence",
        },
        "hot_path": hot_delta,
        "claim_boundary": (
            "observed_cpu_cuda_device_and_latency_delta_only_"
            "not_cuda_speedup_claim_without_repeated_parity_runs"
        ),
    }
    if output_path is not None:
        _write_json(output_path, summary)
    return summary


def run_service_benchmark_device_comparison(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    configure_local_source: bool = True,
    local_source_text: str = DEFAULT_LOCAL_SOURCE_TEXT,
    local_source_tick_steps: int = 1,
    local_source_tick_tokens: int = 128,
    local_source_queue_target_tokens: int | None = None,
    local_source_prewarm_on_startup: bool = True,
    local_source_prewarm_wait_seconds: float = 5.0,
    trace_history_limit: int = 32,
    web_dist_dir: str | Path | None = None,
    feed_text: str = DEFAULT_FEED_TEXT,
    query_text: str = DEFAULT_QUERY_TEXT,
    export_limit: int = 3,
) -> dict[str, Any]:
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir = bundle_dir / "cpu"
    cuda_dir = bundle_dir / "cuda"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)
    cpu_path = cpu_dir / "service-benchmark.json"
    cuda_path = cuda_dir / "service-benchmark.json"
    summary_path = bundle_dir / "device-comparison.json"

    with _temporary_env("MARULHO_DEVICE", "cpu"):
        cpu_report = run_service_benchmark(
            checkpoint_path=checkpoint_path,
            output_path=cpu_path,
            configure_local_source=configure_local_source,
            local_source_text=local_source_text,
            local_source_tick_steps=local_source_tick_steps,
            local_source_tick_tokens=local_source_tick_tokens,
            local_source_queue_target_tokens=local_source_queue_target_tokens,
            local_source_prewarm_on_startup=local_source_prewarm_on_startup,
            local_source_prewarm_wait_seconds=local_source_prewarm_wait_seconds,
            trace_history_limit=trace_history_limit,
            trace_dir=cpu_dir / "traces",
            web_dist_dir=web_dist_dir,
            env_root=cpu_dir,
            feed_text=feed_text,
            query_text=query_text,
            export_limit=export_limit,
        )
    with _temporary_env("MARULHO_DEVICE", "cuda"):
        cuda_report = run_service_benchmark(
            checkpoint_path=checkpoint_path,
            output_path=cuda_path,
            configure_local_source=configure_local_source,
            local_source_text=local_source_text,
            local_source_tick_steps=local_source_tick_steps,
            local_source_tick_tokens=local_source_tick_tokens,
            local_source_queue_target_tokens=local_source_queue_target_tokens,
            local_source_prewarm_on_startup=local_source_prewarm_on_startup,
            local_source_prewarm_wait_seconds=local_source_prewarm_wait_seconds,
            trace_history_limit=trace_history_limit,
            trace_dir=cuda_dir / "traces",
            web_dist_dir=web_dist_dir,
            env_root=cuda_dir,
            feed_text=feed_text,
            query_text=query_text,
            export_limit=export_limit,
        )
    summary = compare_service_benchmark_devices(
        cpu_report=cpu_report,
        cuda_report=cuda_report,
        cpu_report_path=cpu_path,
        cuda_report_path=cuda_path,
        output_path=summary_path,
    )
    summary["paths"]["bundle_dir"] = str(bundle_dir)
    summary["paths"]["summary"] = str(summary_path)
    _write_json(summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local MARULHO service endpoint latency in-process.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-before", type=Path)
    parser.add_argument("--compare-after", type=Path)
    parser.add_argument("--accept-baseline-from", type=Path)
    parser.add_argument("--accepted-by", type=str, default="")
    parser.add_argument("--baseline-label", type=str, default="")
    parser.add_argument("--baseline-note", type=str, default="")
    parser.add_argument("--compare-baseline", type=Path)
    parser.add_argument(
        "--run-against-baseline",
        type=Path,
        help="Run a fresh benchmark and compare it against an accepted baseline; --output is treated as a bundle directory.",
    )
    parser.add_argument(
        "--compare-devices",
        action="store_true",
        help="Run paired CPU/CUDA service benchmarks and write a report-only device comparison; --output is treated as a bundle directory.",
    )
    parser.add_argument(
        "--hot-path-regression-tolerance",
        type=float,
        default=DEFAULT_HOT_PATH_REGRESSION_TOLERANCE,
    )
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--web-dist-dir", type=Path, default=None)
    parser.add_argument("--env-root", type=Path, default=None)
    parser.add_argument("--feed-text", type=str, default=DEFAULT_FEED_TEXT)
    parser.add_argument("--query-text", type=str, default=DEFAULT_QUERY_TEXT)
    parser.add_argument("--export-limit", type=int, default=3)
    parser.add_argument(
        "--configure-local-source",
        action="store_true",
        help="Create and configure a tiny local file source before measuring endpoints.",
    )
    parser.add_argument(
        "--local-source-tick-steps",
        type=int,
        default=0,
        help="Manual Terminus tick steps to run after local source configuration.",
    )
    parser.add_argument(
        "--local-source-tick-tokens",
        type=int,
        default=128,
        help="Configured source tokens per Terminus tick when --configure-local-source is used.",
    )
    parser.add_argument(
        "--local-source-queue-target-tokens",
        type=int,
        default=None,
        help="Configured ingestion queue target tokens; defaults to --local-source-tick-tokens.",
    )
    parser.add_argument(
        "--disable-local-source-prewarm",
        action="store_true",
        help="Keep configured source queue prewarm out of the service benchmark tick setup.",
    )
    parser.add_argument(
        "--local-source-prewarm-wait-seconds",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for configured local source full-queue readiness before measuring a tick.",
    )
    parser.add_argument(
        "--create-synthetic-checkpoint",
        action="store_true",
        help="Create a tiny deterministic checkpoint at --checkpoint when it does not already exist.",
    )
    parser.add_argument(
        "--profile-trainer-stages",
        action="store_true",
        help="Enable opt-in MarulhoTrainer train_step stage timing in the written benchmark report.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.accept_baseline_from is not None:
        result = create_service_benchmark_accepted_baseline(
            report_path=args.accept_baseline_from,
            output_path=args.output,
            accepted_by=args.accepted_by,
            label=args.baseline_label,
            note=args.baseline_note,
        )
        print(
            json.dumps(
                {"output_path": str(args.output), "status": result["status"], "baseline_id": result["baseline_id"]},
                sort_keys=True,
            )
        )
        return
    if args.compare_baseline is not None:
        if args.compare_after is None:
            parser.error("--compare-baseline requires --compare-after")
        result = compare_service_benchmark_against_accepted_baseline(
            baseline_path=args.compare_baseline,
            after_path=args.compare_after,
            output_path=args.output,
            hot_path_regression_tolerance=args.hot_path_regression_tolerance,
        )
        print(json.dumps({"output_path": str(args.output), "status": result["status"]}, sort_keys=True))
        return
    if args.run_against_baseline is not None:
        if args.checkpoint is None:
            parser.error("--run-against-baseline requires --checkpoint")
        if args.create_synthetic_checkpoint and not args.checkpoint.exists():
            create_tiny_service_benchmark_checkpoint(args.checkpoint)
        result = run_service_benchmark_against_accepted_baseline(
            checkpoint_path=args.checkpoint,
            baseline_path=args.run_against_baseline,
            output_dir=args.output,
            configure_local_source=args.configure_local_source,
            local_source_tick_steps=args.local_source_tick_steps,
            local_source_tick_tokens=args.local_source_tick_tokens,
            local_source_queue_target_tokens=args.local_source_queue_target_tokens,
            local_source_prewarm_on_startup=not bool(args.disable_local_source_prewarm),
            local_source_prewarm_wait_seconds=args.local_source_prewarm_wait_seconds,
            trace_dir=args.trace_dir,
            web_dist_dir=args.web_dist_dir,
            env_root=args.env_root,
            feed_text=args.feed_text,
            query_text=args.query_text,
            export_limit=args.export_limit,
            hot_path_regression_tolerance=args.hot_path_regression_tolerance,
        )
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "status": result["status"],
                    "success": result["success"],
                    "comparison_path": result["paths"]["comparison"],
                },
                sort_keys=True,
            )
        )
        return
    if args.compare_devices:
        if args.checkpoint is None:
            parser.error("--compare-devices requires --checkpoint")
        if args.create_synthetic_checkpoint and not args.checkpoint.exists():
            create_tiny_service_benchmark_checkpoint(args.checkpoint)
        result = run_service_benchmark_device_comparison(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            configure_local_source=args.configure_local_source,
            local_source_tick_steps=args.local_source_tick_steps,
            local_source_tick_tokens=args.local_source_tick_tokens,
            local_source_queue_target_tokens=args.local_source_queue_target_tokens,
            local_source_prewarm_on_startup=not bool(args.disable_local_source_prewarm),
            local_source_prewarm_wait_seconds=args.local_source_prewarm_wait_seconds,
            web_dist_dir=args.web_dist_dir,
            feed_text=args.feed_text,
            query_text=args.query_text,
            export_limit=args.export_limit,
        )
        print(
            json.dumps(
                {
                    "output_path": str(args.output),
                    "status": result["status"],
                    "success": result["success"],
                    "comparison_path": result["paths"]["summary"],
                },
                sort_keys=True,
            )
        )
        return
    if args.compare_before is not None or args.compare_after is not None:
        if args.compare_before is None or args.compare_after is None:
            parser.error("--compare-before and --compare-after must be supplied together")
        result = compare_service_benchmark_report_files(
            before_path=args.compare_before,
            after_path=args.compare_after,
            output_path=args.output,
            hot_path_regression_tolerance=args.hot_path_regression_tolerance,
        )
        print(json.dumps({"output_path": str(args.output), "status": result["status"]}, sort_keys=True))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required when not comparing benchmark reports")
    if args.create_synthetic_checkpoint and not args.checkpoint.exists():
        create_tiny_service_benchmark_checkpoint(args.checkpoint)
    result = run_service_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        configure_local_source=args.configure_local_source,
        local_source_tick_steps=args.local_source_tick_steps,
        local_source_tick_tokens=args.local_source_tick_tokens,
        local_source_queue_target_tokens=args.local_source_queue_target_tokens,
        local_source_prewarm_on_startup=not bool(args.disable_local_source_prewarm),
        local_source_prewarm_wait_seconds=args.local_source_prewarm_wait_seconds,
        trace_dir=args.trace_dir,
        web_dist_dir=args.web_dist_dir,
        env_root=args.env_root,
        feed_text=args.feed_text,
        query_text=args.query_text,
        export_limit=args.export_limit,
        profile_trainer_stages=args.profile_trainer_stages,
    )
    print(json.dumps({"output_path": result["output_path"], "success": result["success"]}, sort_keys=True))


if __name__ == "__main__":
    main()
