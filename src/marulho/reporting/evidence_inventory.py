from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


SURFACE = "marulho_evidence_report_inventory.v1"
REPORT_SUMMARY_SURFACE = "marulho_evidence_report_summary.v1"


def build_evidence_report_inventory(
    reports_root: str | Path,
    *,
    limit: int = 20,
    max_report_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Summarize saved JSON reports without running benchmarks or mutating state."""

    root = Path(reports_root)
    root_resolved = root.resolve()
    bounded_limit = max(1, int(limit))
    records: list[dict[str, Any]] = []
    candidates = _json_report_candidates(root_resolved)
    for path in candidates[:bounded_limit]:
        records.append(
            _summarize_report(
                path,
                reports_root=root_resolved,
                max_report_bytes=max(1, int(max_report_bytes)),
            )
        )
    return {
        "surface": SURFACE,
        "reports_root": str(root),
        "reports_root_resolved": str(root_resolved),
        "report_count": len(records),
        "scanned_json_count": len(candidates),
        "limit": bounded_limit,
        "reports_not_run_by_service": True,
        "mutates_runtime_state": False,
        "claim_boundary": (
            "read_only_saved_report_inventory; report presence is evidence routing, "
            "not runtime promotion by itself"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": records,
    }


def _json_report_candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for path in root.rglob("*.json"):
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except Exception:
            continue
        if resolved.is_file():
            paths.append(resolved)
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)


def _summarize_report(
    path: Path,
    *,
    reports_root: Path,
    max_report_bytes: int,
) -> dict[str, Any]:
    stat = path.stat()
    base: dict[str, Any] = {
        "surface": REPORT_SUMMARY_SURFACE,
        "path": str(path),
        "relative_path": path.relative_to(reports_root).as_posix(),
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "readable": False,
        "parse_error": None,
    }
    if int(stat.st_size) > max_report_bytes:
        return {
            **base,
            "parse_error": "report_too_large_for_inventory",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping):
        return {
            **base,
            "parse_error": "json_report_root_is_not_object",
        }
    promotion_gate = _mapping(payload.get("promotion_gate"))
    return {
        **base,
        "readable": True,
        "artifact_kind": _string_or_none(payload.get("artifact_kind") or payload.get("benchmark")),
        "report_surface": _string_or_none(payload.get("surface")),
        "success": _bool_or_none(payload.get("success")),
        "report_status": _string_or_none(payload.get("report_status")),
        "promotion_status": _string_or_none(
            promotion_gate.get("status")
            or payload.get("promotion_status")
            or payload.get("gate_status")
        ),
        "promotes_runtime_claim": _bool_or_none(
            promotion_gate.get("promotes_runtime_claim")
            if "promotes_runtime_claim" in promotion_gate
            else payload.get("promotes_runtime_claim")
        ),
        "target_tokens": _int_or_none(payload.get("target_tokens")),
        "token_delta": _int_or_none(payload.get("token_delta")),
        "tokens_per_second": _float_or_none(payload.get("tokens_per_second")),
        "runtime_owner": _string_or_none(payload.get("runtime_owner")),
        "active_language_path": _string_or_none(payload.get("active_language_path")),
        "external_llm_used": _bool_or_none(payload.get("external_llm_used")),
        "thought_loop_used": _bool_or_none(payload.get("thought_loop_used")),
        "cortex_used": _bool_or_none(payload.get("cortex_used")),
        "missing_required_category_names": _string_list(
            promotion_gate.get("missing_required_category_names")
            or payload.get("missing_required_category_names")
        ),
        "failed_category_names": _string_list(
            promotion_gate.get("failed_category_names")
            or payload.get("failed_category_names")
        ),
        "device": _device_summary(payload),
        "evidence_level": _evidence_level(payload, promotion_gate),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _device_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    device_backend = _mapping(payload.get("device_backend"))
    runtime_device = _mapping(payload.get("runtime_device"))
    brain_status = _mapping(payload.get("brain_status"))
    last_trace = _mapping(brain_status.get("last_trace"))
    return {
        "backend": _string_or_none(
            device_backend.get("backend")
            or runtime_device.get("resolved_device")
            or last_trace.get("executor")
        ),
        "device": _string_or_none(
            device_backend.get("device")
            or runtime_device.get("resolved_device")
            or last_trace.get("device")
        ),
        "cuda_available": _bool_or_none(
            runtime_device.get("cuda_available")
            if "cuda_available" in runtime_device
            else brain_status.get("cuda_available")
        ),
        "promoted_hot_path": _bool_or_none(device_backend.get("promoted_hot_path")),
    }


def _evidence_level(
    payload: Mapping[str, Any],
    promotion_gate: Mapping[str, Any],
) -> str:
    if promotion_gate.get("promotes_runtime_claim") is True:
        return "promotion_claim"
    target_tokens = _int_or_none(payload.get("target_tokens")) or 0
    token_delta = _int_or_none(payload.get("token_delta")) or 0
    success = payload.get("success") is True
    if success and token_delta >= 131_072 and target_tokens >= 131_072:
        return "long_run_evidence"
    if success and token_delta >= 8_192 and target_tokens >= 8_192:
        return "diagnostic_evidence"
    if payload.get("report_status") in {"final", "partial", "timeout", "exception", "interrupt"}:
        return "runtime_report"
    if promotion_gate:
        return "promotion_gate_inventory"
    return "saved_report_inventory"
