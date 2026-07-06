from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


SURFACE = "marulho_evidence_report_inventory.v1"
REPORT_SUMMARY_SURFACE = "marulho_evidence_report_summary.v1"
CURRENT_LANGUAGE_EVIDENCE_SURFACE = "marulho_current_language_evidence_projection.v1"

LANGUAGE_BENCHMARK_SUITE_ARTIFACT = "marulho_language_runtime_benchmark_suite"
LANGUAGE_BRAIN_GENERATION_ARTIFACT = "marulho_language_brain_installed_generation_evidence"
LANGUAGE_BRAIN_GENERATION_REPAIR_ARTIFACT = (
    "marulho_language_brain_installed_generation_repair_evidence"
)
LANGUAGE_BRAIN_GENERATION_REPAIR_SWEEP_ARTIFACT = (
    "marulho_language_brain_installed_generation_repair_sweep"
)


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


def build_current_language_evidence_projection(
    reports_root: str | Path,
    *,
    max_report_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Project current language evidence from saved JSON reports only."""

    root = Path(reports_root)
    root_resolved = root.resolve()
    reports, scan = _load_json_reports(
        root_resolved,
        max_report_bytes=max(1, int(max_report_bytes)),
    )
    suite_entry = _latest_report(reports, LANGUAGE_BENCHMARK_SUITE_ARTIFACT)
    generation_entry = _latest_report(reports, LANGUAGE_BRAIN_GENERATION_ARTIFACT)
    repair_entry = _latest_report(reports, LANGUAGE_BRAIN_GENERATION_REPAIR_ARTIFACT)
    repair_sweep_entry = _latest_report(
        reports,
        LANGUAGE_BRAIN_GENERATION_REPAIR_SWEEP_ARTIFACT,
    )

    suite = suite_entry[1] if suite_entry is not None else {}
    gate = _mapping(suite.get("promotion_gate"))
    generation_category = _category(suite, "generation_coherence")
    generation_evidence = _generation_evidence(
        generation_category,
        generation_entry[1] if generation_entry is not None else {},
        suite,
    )
    repair_evidence = _repair_evidence(
        generation_category,
        repair_entry[1] if repair_entry is not None else {},
        repair_sweep_entry[1] if repair_sweep_entry is not None else {},
    )
    long_run_category = _category(suite, "long_run_throughput")
    throughput_evidence = _throughput_evidence(
        generation_category,
        long_run_category,
        repair_sweep_entry[1] if repair_sweep_entry is not None else {},
    )
    checkpoint_evidence = _current_checkpoint_evidence(
        generation_evidence,
        repair_evidence,
        throughput_evidence,
    )
    gpu_evidence = _gpu_kernel_evidence(
        _category(suite, "gpu_kernel_correctness"),
        throughput_evidence,
    )
    source_entries = [
        ("benchmark_suite", suite_entry),
        ("installed_generation", generation_entry),
        ("installed_generation_repair", repair_entry),
        ("installed_generation_repair_sweep", repair_sweep_entry),
    ]
    source_reports = [
        _source_report_ref(role, entry, root_resolved)
        for role, entry in source_entries
        if entry is not None
    ]

    external_llm_used = _first_bool(
        generation_evidence.get("external_llm_used"),
        repair_evidence.get("external_llm_used"),
        suite.get("external_llm_used"),
        False,
    )
    promotes_runtime_claim = _first_bool(
        gate.get("promotes_runtime_claim"),
        generation_evidence.get("promotes_runtime_claim"),
        repair_evidence.get("promotes_runtime_claim"),
        throughput_evidence.get("promotes_runtime_claim"),
        False,
    )
    promotes_generation_quality_claim = _first_bool(
        generation_evidence.get("promotes_generation_quality_claim"),
        repair_evidence.get("promotes_generation_quality_claim"),
        False,
    )

    return {
        "surface": CURRENT_LANGUAGE_EVIDENCE_SURFACE,
        "reports_root": str(root),
        "reports_root_resolved": str(root_resolved),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only_projection": True,
        "reports_not_run_by_service": True,
        "mutates_runtime_state": False,
        "service_owned_cognition": False,
        "claim_boundary": (
            "saved_report_projection_only; this endpoint does not run benchmarks, "
            "mutate MarulhoBrain, or promote runtime/generation claims by itself"
        ),
        "throughput_baselines": {
            "diagnostic_tokens": 8_192,
            "long_run_gate_tokens": 131_072,
            "house_scale_target_tokens": 524_288,
            "preferred_current_baseline_tokens": 524_288,
        },
        "report_scan": scan,
        "runtime_review_gate": {
            "surface": "marulho_current_language_runtime_review_gate.v1",
            "status": _string_or_none(gate.get("status")),
            "ready_for_review": gate.get("status") == "ready_for_review",
            "missing_required_category_names": _string_list(
                gate.get("missing_required_category_names")
            ),
            "failed_category_names": _string_list(gate.get("failed_category_names")),
            "long_run_evidence_available": _bool_or_none(
                gate.get("long_run_evidence_available")
            ),
            "controlled_decode_house_scale_evidence_available": _bool_or_none(
                gate.get("controlled_decode_house_scale_evidence_available")
            ),
            "generation_controlled_decode_house_scale_aligned": _bool_or_none(
                gate.get("generation_controlled_decode_house_scale_aligned")
            ),
            "brain_installed_generation_evidence_available": _bool_or_none(
                gate.get("brain_installed_generation_evidence_available")
            ),
            "brain_installed_generation_repair_evidence_available": _bool_or_none(
                gate.get("brain_installed_generation_repair_evidence_available")
            ),
            "promotes_runtime_claim": promotes_runtime_claim,
        },
        "current_checkpoint": checkpoint_evidence,
        "generation_evidence": generation_evidence,
        "repair_evidence": repair_evidence,
        "house_scale_throughput_evidence": throughput_evidence,
        "gpu_kernel_evidence": gpu_evidence,
        "ownership": {
            "surface": "marulho_current_language_evidence_ownership.v1",
            "runtime_owner": _first_string(
                repair_evidence.get("runtime_owner"),
                generation_evidence.get("runtime_owner"),
                throughput_evidence.get("runtime_owner"),
            ),
            "service_owner": "thin_brain_adapter",
            "service_owned_cognition": False,
            "external_llm_used": external_llm_used,
            "reports_not_run_by_service": True,
            "mutates_runtime_state": False,
        },
        "capability_claims": {
            "promotes_runtime_claim": promotes_runtime_claim,
            "promotes_generation_quality_claim": promotes_generation_quality_claim,
            "requires_operator_review": True,
        },
        "source_reports": source_reports,
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


def _load_json_reports(
    root: Path,
    *,
    max_report_bytes: int,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], dict[str, Any]]:
    candidates = _json_report_candidates(root)
    records: list[tuple[Path, Mapping[str, Any]]] = []
    skipped_large = 0
    parse_errors = 0
    for path in candidates:
        try:
            if int(path.stat().st_size) > max_report_bytes:
                skipped_large += 1
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            parse_errors += 1
            continue
        if isinstance(payload, Mapping):
            records.append((path, payload))
        else:
            parse_errors += 1
    return records, {
        "surface": "marulho_current_language_evidence_report_scan.v1",
        "scanned_json_count": len(candidates),
        "parsed_report_count": len(records),
        "skipped_large_report_count": skipped_large,
        "parse_error_count": parse_errors,
        "max_report_bytes": int(max_report_bytes),
    }


def _latest_report(
    reports: list[tuple[Path, Mapping[str, Any]]],
    artifact_kind: str,
) -> tuple[Path, Mapping[str, Any]] | None:
    for path, payload in reports:
        if payload.get("artifact_kind") == artifact_kind:
            return path, payload
    return None


def _category(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    categories = report.get("categories")
    if not isinstance(categories, list):
        return {}
    for item in categories:
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    return {}


def _generation_evidence(
    generation_category: Mapping[str, Any],
    generation_report: Mapping[str, Any],
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    category_evidence = _mapping(generation_category.get("evidence"))
    saved_evidence = _mapping(
        category_evidence.get("brain_installed_generation_evidence")
    )
    best = _mapping(saved_evidence.get("best_report"))
    gate = _mapping(generation_report.get("promotion_gate"))
    source = (
        "benchmark_suite.brain_installed_generation_evidence"
        if best
        else "saved_generation_report"
        if generation_report
        else None
    )
    checkpoint_path = _first_string(
        best.get("brain_checkpoint_path"),
        generation_report.get("brain_checkpoint_path"),
        _mapping(generation_report.get("brain_checkpoint")).get("path"),
    )
    case_count = _first_int(best.get("case_count"), gate.get("case_count"))
    passed_count = _first_int(
        best.get("passed_case_count"),
        gate.get("passed_case_count"),
    )
    return {
        "surface": "marulho_current_language_generation_evidence.v1",
        "available": bool(best or generation_report),
        "source": source,
        "active_language_path": _first_string(
            best.get("active_language_path"),
            generation_report.get("active_language_path"),
            suite.get("active_language_path"),
        ),
        "runtime_owner": _first_string(
            best.get("runtime_owner"),
            generation_report.get("runtime_owner"),
        ),
        "brain_checkpoint_path": checkpoint_path,
        "brain_checkpoint_restore_verified": _first_bool(
            best.get("brain_checkpoint_restore_verified"),
            gate.get("brain_checkpoint_restore_verified"),
        ),
        "case_count": case_count,
        "passed_case_count": passed_count,
        "case_pass_rate": _first_float(best.get("case_pass_rate")),
        "mean_prefix_match_chars": _first_float(best.get("mean_prefix_match_chars")),
        "mean_prefix_match_fraction": _first_float(
            best.get("mean_prefix_match_fraction")
        ),
        "generation_runs_through_marulho_brain": _first_bool(
            best.get("generation_runs_through_marulho_brain"),
            gate.get("generation_runs_through_marulho_brain"),
        ),
        "grounded_prompt_suite_available": _first_bool(
            best.get("grounded_prompt_suite_available"),
            gate.get("grounded_prompt_suite_available"),
        ),
        "status_read_mutation_absent": _first_bool(
            best.get("status_read_mutation_absent"),
            generation_report.get("status_read_mutation") is False
            if "status_read_mutation" in generation_report
            else None,
        ),
        "external_llm_used": _first_bool(
            generation_report.get("external_llm_used"),
            suite.get("external_llm_used"),
            False,
        ),
        "service_owned_cognition": _first_bool(
            generation_report.get("service_owned_cognition"),
            False,
        ),
        "promotes_runtime_claim": _first_bool(
            best.get("promotes_runtime_claim"),
            gate.get("promotes_runtime_claim"),
            generation_report.get("promotes_runtime_claim"),
            False,
        ),
        "promotes_generation_quality_claim": _first_bool(
            best.get("promotes_generation_quality_claim"),
            gate.get("promotes_generation_quality_claim"),
            generation_report.get("promotes_generation_quality_claim"),
            False,
        ),
    }


def _repair_evidence(
    generation_category: Mapping[str, Any],
    repair_report: Mapping[str, Any],
    repair_sweep: Mapping[str, Any],
) -> dict[str, Any]:
    category_evidence = _mapping(generation_category.get("evidence"))
    saved_evidence = _mapping(
        category_evidence.get("brain_installed_generation_repair_evidence")
    )
    saved_best = _mapping(saved_evidence.get("best_report"))
    selected = _mapping(repair_sweep.get("selected_repair_evidence"))
    selection = _mapping(repair_sweep.get("candidate_selection"))
    gate = _mapping(repair_sweep.get("promotion_gate"))
    repair_gate = _mapping(repair_report.get("promotion_gate"))
    best = selected or saved_best or repair_report
    source = (
        "generation_repair_sweep.selected_repair_evidence"
        if selected
        else "benchmark_suite.brain_installed_generation_repair_evidence"
        if saved_best
        else "saved_generation_repair_report"
        if repair_report
        else None
    )
    return {
        "surface": "marulho_current_language_generation_repair_evidence.v1",
        "available": bool(best),
        "source": source,
        "runtime_owner": _first_string(
            repair_sweep.get("runtime_owner"),
            saved_best.get("runtime_owner"),
            repair_report.get("runtime_owner"),
        ),
        "active_language_path": _first_string(
            repair_sweep.get("active_language_path"),
            best.get("active_language_path"),
            repair_report.get("active_language_path"),
        ),
        "candidate_count": _first_int(
            selection.get("candidate_count"),
            repair_sweep.get("candidate_count"),
            gate.get("candidate_count"),
        ),
        "selected_candidate_id": _first_string(
            selection.get("selected_candidate_id"),
            gate.get("selected_candidate_id"),
            best.get("candidate_id"),
        ),
        "selected_checkpoint_path": _first_string(
            selection.get("selected_repaired_brain_checkpoint_path"),
            best.get("repaired_brain_checkpoint_path"),
            repair_report.get("repaired_brain_checkpoint_path"),
        ),
        "selected_checkpoint_sha256": _first_string(
            selection.get("selected_repaired_brain_checkpoint_sha256"),
            best.get("repaired_brain_checkpoint_sha256"),
        ),
        "checkpoint_restore_verified": _first_bool(
            gate.get("selected_checkpoint_restore_verified"),
            best.get("repaired_brain_checkpoint_restore_verified"),
            repair_gate.get("repaired_brain_checkpoint_restore_verified"),
        ),
        "pre_passed_case_count": _first_int(best.get("pre_passed_case_count")),
        "post_passed_case_count": _first_int(
            best.get("post_passed_case_count"),
            gate.get("selected_post_passed_case_count"),
        ),
        "case_count": _first_int(best.get("case_count"), gate.get("selected_case_count")),
        "passed_case_count_delta": _first_int(best.get("passed_case_count_delta")),
        "mean_prefix_match_chars_delta": _first_float(
            best.get("mean_prefix_match_chars_delta")
        ),
        "regressed_prompt_count": _first_int(
            best.get("regressed_prompt_count"),
            gate.get("selected_regressed_prompt_count"),
        ),
        "update_token_count": _first_int(
            best.get("update_token_count"),
            gate.get("selected_update_token_count"),
        ),
        "tokens_per_second": _first_float(
            best.get("tokens_per_second"),
            gate.get("selected_tokens_per_second"),
        ),
        "total_window_tokens_per_second": _first_float(
            best.get("total_window_tokens_per_second")
        ),
        "learning_config": dict(_mapping(best.get("learning_config"))),
        "external_llm_used": _first_bool(repair_sweep.get("external_llm_used"), False),
        "service_owned_cognition": _first_bool(
            repair_sweep.get("service_owned_cognition"),
            False,
        ),
        "promotes_runtime_claim": _first_bool(
            repair_sweep.get("promotes_runtime_claim"),
            gate.get("promotes_runtime_claim"),
            saved_best.get("promotes_runtime_claim"),
            False,
        ),
        "promotes_generation_quality_claim": _first_bool(
            repair_sweep.get("promotes_generation_quality_claim"),
            gate.get("promotes_generation_quality_claim"),
            saved_best.get("promotes_generation_quality_claim"),
            False,
        ),
    }


def _throughput_evidence(
    generation_category: Mapping[str, Any],
    long_run_category: Mapping[str, Any],
    repair_sweep: Mapping[str, Any],
) -> dict[str, Any]:
    post_repair = _mapping(repair_sweep.get("post_repair_sustained_window"))
    category_evidence = _mapping(generation_category.get("evidence"))
    alignment = _mapping(
        category_evidence.get("brain_installed_generation_long_run_alignment")
    )
    matching = alignment.get("matching_reports")
    alignment_reports = [item for item in matching if isinstance(item, Mapping)] if isinstance(matching, list) else []
    long_run_evidence = _mapping(long_run_category.get("evidence"))
    best_long = _mapping(long_run_evidence.get("best_long_report"))
    if not best_long and alignment_reports:
        best_long = max(
            alignment_reports,
            key=lambda item: (
                _first_int(item.get("token_delta")) or 0,
                _first_float(item.get("tokens_per_second")) or 0.0,
            ),
        )
    report = post_repair or best_long
    target_tokens = _first_int(report.get("target_tokens"))
    token_delta = _first_int(report.get("token_delta"))
    tracked_triton_kernel_names = _string_list(
        report.get("tracked_triton_kernel_used_names")
    )
    house_scale_reached = bool(
        (token_delta or 0) >= 524_288
        and (target_tokens or 0) >= 524_288
        and report.get("success") is True
    )
    return {
        "surface": "marulho_current_language_house_scale_throughput_evidence.v1",
        "available": bool(report),
        "source": (
            "generation_repair_sweep.post_repair_sustained_window"
            if post_repair
            else "benchmark_suite.long_run_alignment"
            if best_long
            else None
        ),
        "runtime_owner": _first_string(report.get("runtime_owner")),
        "active_language_path": _first_string(report.get("active_language_path")),
        "checkpoint_path": _first_string(
            report.get("checkpoint_path"),
            alignment.get("generation_checkpoint_path"),
        ),
        "backend": _first_string(report.get("backend"), report.get("mode")),
        "device": _first_string(report.get("device")),
        "success": _bool_or_none(report.get("success")),
        "report_status": _string_or_none(report.get("report_status")),
        "target_tokens": target_tokens,
        "token_delta": token_delta,
        "tokens_per_second": _first_float(report.get("tokens_per_second")),
        "diagnostic_boundary_reached": _first_bool(
            report.get("diagnostic_boundary_reached"),
            (token_delta or 0) >= 8_192 and (target_tokens or 0) >= 8_192,
        ),
        "long_run_gate_reached": _first_bool(
            report.get("long_run_gate_reached"),
            (token_delta or 0) >= 131_072 and (target_tokens or 0) >= 131_072,
        ),
        "house_scale_gate_reached": _first_bool(
            report.get("house_scale_gate_reached"),
            house_scale_reached,
        ),
        "controlled_decode_house_scale_aligned": _bool_or_none(
            alignment.get("same_checkpoint_controlled_decode_house_scale_available")
        ),
        "triton_kernel_used": _first_bool(
            report.get("triton_kernel_used"),
            bool(tracked_triton_kernel_names) if tracked_triton_kernel_names else None,
        ),
        "tracked_triton_kernel_failure_count": _first_int(
            report.get("tracked_triton_kernel_failure_count")
        ),
        "tracked_triton_kernel_used_names": tracked_triton_kernel_names,
        "generation_decode": dict(_mapping(report.get("generation_decode"))),
        "promotes_hot_path": _first_bool(
            report.get("promotes_hot_path"),
            report.get("promoted_hot_path"),
            False,
        ),
        "promotes_runtime_claim": _first_bool(
            report.get("promotes_runtime_claim"),
            False,
        ),
    }


def _current_checkpoint_evidence(
    generation: Mapping[str, Any],
    repair: Mapping[str, Any],
    throughput: Mapping[str, Any],
) -> dict[str, Any]:
    path = _first_string(
        repair.get("selected_checkpoint_path"),
        generation.get("brain_checkpoint_path"),
        throughput.get("checkpoint_path"),
    )
    return {
        "surface": "marulho_current_language_checkpoint_evidence.v1",
        "path": path,
        "sha256": _first_string(repair.get("selected_checkpoint_sha256")),
        "source": (
            "generation_repair_selected_checkpoint"
            if repair.get("selected_checkpoint_path")
            else "installed_generation_checkpoint"
            if generation.get("brain_checkpoint_path")
            else "house_scale_throughput_checkpoint"
            if throughput.get("checkpoint_path")
            else None
        ),
        "restore_verified": _first_bool(
            repair.get("checkpoint_restore_verified"),
            generation.get("brain_checkpoint_restore_verified"),
        ),
        "delete_protected_by_current_evidence": bool(path),
    }


def _gpu_kernel_evidence(
    gpu_category: Mapping[str, Any],
    throughput: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _mapping(gpu_category.get("evidence"))
    covered = _string_list(evidence.get("covered_kernel_names"))
    tracked = _string_list(throughput.get("tracked_triton_kernel_used_names"))
    return {
        "surface": "marulho_current_language_gpu_kernel_evidence.v1",
        "available": bool(covered or tracked),
        "category_passed": _bool_or_none(gpu_category.get("passed")),
        "covered_kernel_count": len(covered),
        "covered_kernel_names": covered,
        "generation_triton_kernel_used": _bool_or_none(
            throughput.get("triton_kernel_used")
        ),
        "generation_tracked_kernel_names": tracked,
        "generation_tracked_failure_count": _first_int(
            throughput.get("tracked_triton_kernel_failure_count")
        ),
    }


def _source_report_ref(
    role: str,
    entry: tuple[Path, Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    path, payload = entry
    stat = path.stat()
    return {
        "surface": "marulho_current_language_source_report_ref.v1",
        "role": role,
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "artifact_kind": _string_or_none(payload.get("artifact_kind")),
        "report_surface": _string_or_none(payload.get("surface")),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


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


def _first_string(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value) != "":
            return str(value)
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        result = _int_or_none(value)
        if result is not None:
            return result
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        result = _float_or_none(value)
        if result is not None:
            return result
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
