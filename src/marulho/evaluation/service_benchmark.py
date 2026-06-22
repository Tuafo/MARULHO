from __future__ import annotations

import argparse
from collections import Counter
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
SETUP_ENDPOINTS = frozenset({"terminus_configure", "terminus_tick"})
HOT_PATH_ENDPOINTS = frozenset({"feed", "query", "respond"})
STATUS_ENDPOINTS = frozenset({"health", "status", "terminus", "living_loop", "policy_actuator"})
SLOW_PATH_ENDPOINTS = frozenset(
    {
        "replay_plan",
        "replay_sample_history",
        "export",
        "replay_dataset_preview",
        "replay_dataset_bundle",
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
        title="MARULHO Service Benchmark",
    )


def _load_json_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark report must be a JSON object: {path}")
    return payload


def _runtime_truth_verdict_from_report(report: Mapping[str, Any]) -> str:
    for key in ("status_runtime_truth_summary", "terminus_runtime_truth_summary", "runtime_truth"):
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
        "slow_path_note": "Replay, export, bundle, and dataset endpoints are explicit tooling/evaluation surfaces, not always-on runtime work.",
    }
    return summary


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


def _summarize_runtime_device_evidence(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
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


def _summarize_replay_sample_history(history_body: Any, fallback: Any = None) -> dict[str, Any] | None:
    if isinstance(history_body, dict):
        history = history_body.get("history")
    else:
        history = None
    records = [dict(item) for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    if not records and isinstance(fallback, dict):
        existing = fallback.get("replay_sample_summary")
        if isinstance(existing, dict):
            return dict(existing)
    if not records and not isinstance(history_body, dict):
        return None

    mode_counts: Counter[str] = Counter({"dry_run": 0, "sample": 0})
    status_counts: Counter[str] = Counter()
    selected_count = 0
    for record in records:
        mode = str(record.get("mode") or "sample")
        if mode not in {"dry_run", "sample"}:
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
        "history_endpoint": "/terminus/replay-sample/history",
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
    include_living_loop_telemetry: bool = True,
    profile_trainer_stages: bool = False,
) -> dict[str, Any]:
    """Measure the local service endpoints in-process and write JSON results."""
    started = time.perf_counter()
    endpoint_timings: list[dict[str, Any]] = []
    response_bodies: dict[str, Any] = {}
    configured_source_warmup_evidence: dict[str, Any] | None = None
    manager = getattr(getattr(app, "state", None), "marulho_manager", None)
    trainer = getattr(manager, "_trainer", None)
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
            queue_target_tokens = max(
                tick_tokens,
                int(
                    configured_source_queue_target_tokens
                    if configured_source_queue_target_tokens is not None
                    else tick_tokens
                ),
            )
            configure_record, configure_body = _measure_endpoint(
                client,
                name="terminus_configure",
                method="POST",
                path="/terminus/configure",
                json_body={
                    "source_bank": [
                        {
                            "name": str(configured_source_name),
                            "source": str(Path(configured_source_path)),
                            "source_type": "file",
                        }
                    ],
                    "tick_tokens": tick_tokens,
                    "sleep_interval_seconds": 0.01,
                    "repeat_sources": True,
                    "ingestion": {
                        "enabled": True,
                        "queue_target_tokens": queue_target_tokens,
                        "prewarm_on_startup": bool(configured_source_prewarm_on_startup),
                        "prewarm_max_seconds": 0.05,
                    },
                },
            )
            endpoint_timings.append(configure_record)
            response_bodies["terminus_configure"] = configure_body
            if bool(configured_source_prewarm_on_startup):
                warmup_started = time.perf_counter()
                deadline = warmup_started + max(0.0, float(configured_source_prewarm_wait_seconds))
                attempts = 0
                last_runtime: Mapping[str, Any] = {}
                while True:
                    attempts += 1
                    poll_record, poll_body = _measure_endpoint(
                        client,
                        name="terminus_prewarm_poll",
                        method="GET",
                        path="/terminus",
                    )
                    endpoint_timings.append(poll_record)
                    if isinstance(poll_body, Mapping):
                        runtime_body = poll_body.get("terminus_runtime")
                        if isinstance(runtime_body, Mapping):
                            last_runtime = runtime_body
                    ingestion = (
                        last_runtime.get("ingestion")
                        if isinstance(last_runtime.get("ingestion"), Mapping)
                        else {}
                    )
                    if bool(ingestion.get("full_warm_ready", False)):
                        break
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(0.01)
                ingestion = (
                    last_runtime.get("ingestion")
                    if isinstance(last_runtime.get("ingestion"), Mapping)
                    else {}
                )
                configured_source_warmup_evidence = {
                    "enabled": True,
                    "mode": "prewarm_before_measured_tick",
                    "not_hot_path": True,
                    "attempts": int(attempts),
                    "wait_duration_ms": float((time.perf_counter() - warmup_started) * 1000.0),
                    "wait_budget_seconds": float(configured_source_prewarm_wait_seconds),
                    "warm_ready": bool(ingestion.get("warm_ready", False)),
                    "full_warm_ready": bool(ingestion.get("full_warm_ready", False)),
                    "ready_source_count": int(ingestion.get("ready_source_count", 0) or 0),
                    "full_queue_source_count": int(ingestion.get("full_queue_source_count", 0) or 0),
                    "total_buffered_tokens": int(ingestion.get("total_buffered_tokens", 0) or 0),
                    "queue_target_tokens": int(ingestion.get("queue_target_tokens", queue_target_tokens) or queue_target_tokens),
                    "prewarm_last_duration_ms": ingestion.get("prewarm_last_duration_ms"),
                    "startup_warm_latency_ms": ingestion.get("startup_warm_latency_ms"),
                }
                response_bodies["terminus_prewarm_poll"] = {
                    "terminus_runtime": dict(last_runtime),
                    "warmup_evidence": dict(configured_source_warmup_evidence),
                }
            if int(configured_source_tick_steps) > 0:
                if profile_configured_tick_only and trainer is not None:
                    enable_profile = getattr(trainer, "enable_train_step_profile", None)
                    if callable(enable_profile):
                        enable_profile(reset=True)
                tick_record, tick_body = _measure_endpoint(
                    client,
                    name="terminus_tick",
                    method="POST",
                    path="/terminus/tick",
                    json_body={"steps": int(configured_source_tick_steps)},
                )
                if profile_configured_tick_only and trainer is not None:
                    report_profile = getattr(trainer, "train_step_profile_report", None)
                    if callable(report_profile):
                        captured_trainer_stage_profile = dict(report_profile())
                    disable_profile = getattr(trainer, "disable_train_step_profile", None)
                    if callable(disable_profile):
                        disable_profile()
                endpoint_timings.append(tick_record)
                response_bodies["terminus_tick"] = tick_body

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
    status_device_evidence = _summarize_runtime_device_evidence(response_bodies.get("status"))
    terminus_device_evidence = _summarize_runtime_device_evidence(response_bodies.get("terminus"))
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

    configured_source_summary = None
    if configured_source_path is not None:
        configure_body = response_bodies.get("terminus_configure")
        tick_body = response_bodies.get("terminus_tick")
        runtime = (
            configure_body.get("terminus_runtime")
            if isinstance(configure_body, dict) and isinstance(configure_body.get("terminus_runtime"), dict)
            else {}
        )
        tick_runtime = (
            tick_body.get("terminus_runtime")
            if isinstance(tick_body, dict) and isinstance(tick_body.get("terminus_runtime"), dict)
            else {}
        )
        tick_summaries = (
            list(tick_body.get("tick_summaries") or [])
            if isinstance(tick_body, dict) and isinstance(tick_body.get("tick_summaries"), list)
            else []
        )
        latest_tick_summary = (
            tick_summaries[-1]
            if tick_summaries and isinstance(tick_summaries[-1], dict)
            else {}
        )
        tick_source = (
            latest_tick_summary.get("source")
            if isinstance(latest_tick_summary.get("source"), dict)
            else {}
        )
        tick_concept_observation = (
            tick_source.get("concept_observation")
            if isinstance(tick_source.get("concept_observation"), dict)
            else {}
        )
        tick_stage_timings = (
            latest_tick_summary.get("stage_timings_ms")
            if isinstance(latest_tick_summary.get("stage_timings_ms"), dict)
            else {}
        )
        source_progress = (
            list(tick_runtime.get("source_progress") or [])
            if isinstance(tick_runtime.get("source_progress"), list)
            else []
        )
        first_source_progress = (
            source_progress[0]
            if source_progress and isinstance(source_progress[0], dict)
            else {}
        )
        source_cache_summary = {
            "cache_write_count": int(first_source_progress.get("cache_write_count", 0) or 0),
            "cache_schedule_count": int(first_source_progress.get("cache_schedule_count", 0) or 0),
            "cache_skip_count": int(first_source_progress.get("cache_skip_count", 0) or 0),
            "cache_failure_count": int(first_source_progress.get("cache_failure_count", 0) or 0),
            "cache_pending": bool(first_source_progress.get("cache_pending", False)),
            "last_cache_update_mode": str(first_source_progress.get("last_cache_update_mode", "not_run") or "not_run"),
        }
        configured_source_summary = {
            "enabled": True,
            "source_name": str(configured_source_name),
            "source_path": str(Path(configured_source_path)),
            "tick_steps": int(configured_source_tick_steps),
            "configure_success": bool(response_bodies.get("terminus_configure") is not None),
            "tick_success": bool(response_bodies.get("terminus_tick") is not None) if int(configured_source_tick_steps) > 0 else None,
            "configured": bool(runtime.get("configured")),
            "source_count": int(runtime.get("source_count", 0) or 0),
            "tick_tokens_processed": int(tick_runtime.get("last_tick_token_delta", 0) or 0),
            "background_tokens_processed": int(tick_runtime.get("background_tokens_processed", 0) or 0),
            "last_tick_duration_ms": tick_runtime.get("last_tick_duration_ms"),
            "stage_timings_ms": dict(tick_stage_timings),
            "concept_observation": dict(tick_concept_observation),
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
        "runtime_device_evidence": {
            "status": status_device_evidence,
            "terminus": terminus_device_evidence,
            "semantics": {
                "claim_boundary": "observed_device_placement_only_not_cuda_speedup",
                "cuda_claim_requires": "observed_cuda_execution_true_plus_cpu_cuda_parity_or_benchmark_delta",
            },
        },
        "feedback_telemetry": feedback_telemetry,
        "policy_actuator_summary": policy_actuator_summary,
        "replay_plan_summary": replay_plan_summary,
        "replay_sample_summary": replay_sample_summary,
        "trace_export_summary": export_summary,
        "replay_dataset_summary": replay_dataset_summary,
        "replay_dataset_bundle_summary": replay_dataset_bundle_summary,
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
    include_living_loop_telemetry: bool = True,
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
        include_living_loop_telemetry=include_living_loop_telemetry,
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
