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
LANGUAGE_BRAIN_CONTINUAL_LEARNING_ARTIFACT = (
    "marulho_language_brain_installed_continual_learning_evidence"
)
LANGUAGE_BRAIN_STRUCTURAL_PLASTICITY_ARTIFACT = (
    "marulho_language_brain_installed_structural_plasticity_evidence"
)
LANGUAGE_STATE_BLOCK_RUNTIME_IMPACT_ARTIFACT = (
    "marulho_language_state_block_runtime_impact"
)
LANGUAGE_ELIGIBILITY_TRACE_RUNTIME_IMPACT_ARTIFACT = (
    "marulho_language_eligibility_trace_runtime_impact"
)
LANGUAGE_MEMORY_SLOT_TRAINING_IMPACT_ARTIFACT = (
    "marulho_language_memory_slot_training_impact"
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
    continual_learning_entry = _latest_report(
        reports,
        LANGUAGE_BRAIN_CONTINUAL_LEARNING_ARTIFACT,
    )
    structural_plasticity_entry = _latest_report(
        reports,
        LANGUAGE_BRAIN_STRUCTURAL_PLASTICITY_ARTIFACT,
    )
    state_block_impact_entry = _latest_report(
        reports,
        LANGUAGE_STATE_BLOCK_RUNTIME_IMPACT_ARTIFACT,
    )
    eligibility_impact_entry = _latest_report(
        reports,
        LANGUAGE_ELIGIBILITY_TRACE_RUNTIME_IMPACT_ARTIFACT,
    )
    latest_memory_slot_training_impact_entry = _latest_report(
        reports,
        LANGUAGE_MEMORY_SLOT_TRAINING_IMPACT_ARTIFACT,
    )
    memory_slot_training_impact_entry = _latest_complete_report(
        reports,
        LANGUAGE_MEMORY_SLOT_TRAINING_IMPACT_ARTIFACT,
    )

    suite = suite_entry[1] if suite_entry is not None else {}
    gate = _mapping(suite.get("promotion_gate"))
    generation_category = _category(suite, "generation_coherence")
    continual_learning_category = _category(suite, "continual_learning")
    forgetting_category = _category(suite, "forgetting")
    replay_recovery_category = _category(suite, "replay_recovery")
    growth_prune_category = _category(suite, "growth_prune_safety")
    checkpoint_restore_category = _category(suite, "checkpoint_restore")
    rollback_category = _category(suite, "rollback")
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
    training_evidence = _training_throughput_evidence(
        continual_learning_category,
        continual_learning_entry[1] if continual_learning_entry is not None else {},
    )
    forgetting_replay_evidence = _forgetting_replay_evidence(
        forgetting_category,
        replay_recovery_category,
        training_evidence,
    )
    active_compute_evidence = _active_compute_evidence(_category(suite, "active_compute"))
    structural_evidence = _structural_plasticity_evidence(
        growth_prune_category,
        structural_plasticity_entry[1]
        if structural_plasticity_entry is not None
        else {},
    )
    checkpoint_lineage_evidence = _checkpoint_lineage_evidence(
        checkpoint_restore_category,
        rollback_category,
        training_evidence,
        structural_evidence,
        repair_evidence,
    )
    bottleneck_evidence = _backend_bottleneck_evidence(
        training_evidence,
        state_block_impact_entry[1] if state_block_impact_entry is not None else {},
        eligibility_impact_entry[1] if eligibility_impact_entry is not None else {},
        memory_slot_training_impact_entry[1]
        if memory_slot_training_impact_entry is not None
        else {},
        latest_memory_slot_training_impact_entry[1]
        if latest_memory_slot_training_impact_entry is not None
        else {},
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
        ("installed_continual_learning", continual_learning_entry),
        ("installed_structural_plasticity", structural_plasticity_entry),
        ("state_block_runtime_impact", state_block_impact_entry),
        ("eligibility_trace_runtime_impact", eligibility_impact_entry),
        (
            "memory_slot_training_impact_backend_decision",
            memory_slot_training_impact_entry,
        ),
        ("memory_slot_training_impact_latest", latest_memory_slot_training_impact_entry),
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
        "training_throughput_evidence": training_evidence,
        "forgetting_replay_evidence": forgetting_replay_evidence,
        "structural_plasticity_evidence": structural_evidence,
        "checkpoint_lineage_evidence": checkpoint_lineage_evidence,
        "house_scale_throughput_evidence": throughput_evidence,
        "active_compute_evidence": active_compute_evidence,
        "gpu_kernel_evidence": gpu_evidence,
        "backend_bottleneck_evidence": bottleneck_evidence,
        "ownership": {
            "surface": "marulho_current_language_evidence_ownership.v1",
            "runtime_owner": _first_string(
                repair_evidence.get("runtime_owner"),
                generation_evidence.get("runtime_owner"),
                training_evidence.get("runtime_owner"),
                structural_evidence.get("runtime_owner"),
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


def _latest_complete_report(
    reports: list[tuple[Path, Mapping[str, Any]]],
    artifact_kind: str,
) -> tuple[Path, Mapping[str, Any]] | None:
    for path, payload in reports:
        if (
            payload.get("artifact_kind") == artifact_kind
            and _report_is_complete(payload)
        ):
            return path, payload
    return None


def _report_is_complete(payload: Mapping[str, Any]) -> bool:
    status = _string_or_none(payload.get("report_status"))
    if status is None:
        return bool(_mapping(payload.get("comparison")))
    return status == "final"


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


def _training_throughput_evidence(
    continual_learning_category: Mapping[str, Any],
    continual_learning_report: Mapping[str, Any],
) -> dict[str, Any]:
    category_evidence = _mapping(continual_learning_category.get("evidence"))
    saved = _mapping(
        category_evidence.get("brain_installed_continual_learning_evidence")
    )
    saved_best = _mapping(saved.get("best_report"))
    summary = _mapping(continual_learning_report.get("learning_summary"))
    post_learning = _mapping(
        continual_learning_report.get("post_learning_sustained_window")
    )
    gate = _mapping(continual_learning_report.get("promotion_gate"))
    memory_slots = _mapping(
        saved_best.get("memory_slots")
        if isinstance(saved_best.get("memory_slots"), Mapping)
        else summary.get("memory_slots")
    )
    best = saved_best or summary or continual_learning_report
    source = (
        "benchmark_suite.brain_installed_continual_learning_evidence"
        if saved_best
        else "saved_brain_installed_continual_learning_report"
        if continual_learning_report
        else None
    )
    return {
        "surface": "marulho_current_language_training_throughput_evidence.v1",
        "available": bool(best),
        "source": source,
        "runtime_owner": _first_string(
            best.get("runtime_owner"),
            continual_learning_report.get("runtime_owner"),
        ),
        "active_language_path": _first_string(
            best.get("active_language_path"),
            continual_learning_report.get("active_language_path"),
        ),
        "training_surface": _first_string(
            best.get("training_surface"),
            summary.get("training_surface"),
        ),
        "brain_surface": _first_string(
            best.get("brain_surface"),
            summary.get("brain_surface"),
        ),
        "learning_status": _first_string(
            best.get("learning_status"),
            summary.get("status"),
        ),
        "device": _first_string(best.get("device"), summary.get("device")),
        "update_token_count": _first_int(
            best.get("update_token_count"),
            summary.get("update_token_count"),
            continual_learning_report.get("update_token_count"),
        ),
        "house_scale_update_tokens_reached": _first_bool(
            best.get("house_scale_update_tokens_reached"),
            gate.get("house_scale_524288_update_tokens_reached"),
        ),
        "tokens_per_second": _first_float(
            best.get("tokens_per_second"),
            summary.get("tokens_per_second"),
            continual_learning_report.get("tokens_per_second"),
        ),
        "total_window_tokens_per_second": _first_float(
            best.get("total_window_tokens_per_second"),
            summary.get("total_window_tokens_per_second"),
            continual_learning_report.get("total_window_tokens_per_second"),
        ),
        "new_domain_loss_delta": _first_float(
            best.get("new_domain_loss_delta"),
            summary.get("new_domain_loss_delta"),
        ),
        "old_domain_forgetting": _first_float(
            best.get("old_domain_forgetting"),
            summary.get("old_domain_forgetting"),
        ),
        "general_replay_retention_delta": _first_float(
            best.get("general_replay_retention_delta"),
            summary.get("general_replay_retention_delta"),
        ),
        "final_parameter_delta_l2": _first_float(
            best.get("final_parameter_delta_l2"),
            summary.get("final_parameter_delta_l2"),
        ),
        "learned_brain_checkpoint_path": _first_string(
            best.get("learned_brain_checkpoint_path"),
            continual_learning_report.get("learned_brain_checkpoint_path"),
        ),
        "learned_brain_checkpoint_restore_verified": _first_bool(
            best.get("learned_brain_checkpoint_restore_verified"),
            gate.get("learned_brain_checkpoint_restore_verified"),
            _mapping(continual_learning_report.get("learned_brain_checkpoint")).get(
                "restore_verified"
            ),
        ),
        "status_read_mutation_absent": _first_bool(
            best.get("status_read_mutation_absent"),
            gate.get("status_read_mutation_absent"),
            continual_learning_report.get("status_read_mutation") is False
            if "status_read_mutation" in continual_learning_report
            else None,
        ),
        "memory_slots": {
            "surface": "marulho_current_language_training_memory_slot_evidence.v1",
            "enabled": _first_bool(
                memory_slots.get("enabled"),
                best.get("memory_slots_enabled"),
            ),
            "bounded_path": _first_bool(
                memory_slots.get("bounded_memory_slot_path"),
                best.get("memory_slot_bounded_path"),
            ),
            "candidate_slots_scored": _first_int(
                memory_slots.get("candidate_slots_scored"),
                best.get("memory_slot_candidate_slots_scored"),
            ),
            "runs_all_slots": _first_bool(
                memory_slots.get("runs_all_slots"),
                best.get("memory_slot_runs_all_slots"),
            ),
            "retrieval_backend": _first_string(
                memory_slots.get("memory_slot_retrieval_backend"),
                best.get("memory_slot_retrieval_backend"),
            ),
        },
        "training_window_triton_accounting": dict(
            _mapping(
                summary.get("training_window_triton_accounting")
                or best.get("training_window_triton_accounting")
            )
        ),
        "post_learning_sustained": {
            "surface": "marulho_current_language_post_learning_sustained_evidence.v1",
            "enabled": _first_bool(
                post_learning.get("enabled"),
                best.get("post_learning_sustained_enabled"),
            ),
            "success": _first_bool(
                post_learning.get("success"),
                best.get("post_learning_sustained_success"),
            ),
            "target_tokens": _first_int(post_learning.get("target_tokens")),
            "token_delta": _first_int(
                post_learning.get("token_delta"),
                best.get("post_learning_sustained_token_delta"),
            ),
            "tokens_per_second": _first_float(
                post_learning.get("tokens_per_second"),
                best.get("post_learning_sustained_tokens_per_second"),
            ),
            "backend": _first_string(
                post_learning.get("backend"),
                best.get("post_learning_sustained_backend"),
            ),
            "tracked_triton_kernel_failure_count": _first_int(
                post_learning.get("tracked_triton_kernel_failure_count")
            ),
            "tracked_triton_kernel_used_names": _string_list(
                post_learning.get("tracked_triton_kernel_used_names")
                or best.get("post_learning_sustained_triton_kernel_names")
            ),
        },
        "external_llm_used": _first_bool(
            continual_learning_report.get("external_llm_used"),
            False,
        ),
        "service_owned_cognition": _first_bool(
            continual_learning_report.get("service_owned_cognition"),
            False,
        ),
        "promotes_runtime_claim": _first_bool(
            best.get("promotes_runtime_claim"),
            continual_learning_report.get("promotes_runtime_claim"),
            gate.get("promotes_runtime_claim"),
            False,
        ),
    }


def _forgetting_replay_evidence(
    forgetting_category: Mapping[str, Any],
    replay_recovery_category: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    forgetting = _mapping(forgetting_category.get("evidence"))
    replay = _mapping(replay_recovery_category.get("evidence"))
    return {
        "surface": "marulho_current_language_forgetting_replay_evidence.v1",
        "forgetting_measured": _first_bool(
            forgetting.get("brain_installed_forgetting_measured"),
            training.get("old_domain_forgetting") is not None,
        ),
        "old_domain_forgetting": _first_float(
            forgetting.get("brain_installed_old_domain_forgetting"),
            training.get("old_domain_forgetting"),
        ),
        "old_domain_forgetting_within_tolerance": _bool_or_none(
            forgetting.get("old_domain_forgetting_within_tolerance")
        ),
        "replay_retention_measured": _first_bool(
            replay.get("brain_installed_replay_retention_measured"),
            training.get("general_replay_retention_delta") is not None,
        ),
        "general_replay_retention_delta": _first_float(
            replay.get("brain_installed_general_replay_retention_delta"),
            training.get("general_replay_retention_delta"),
        ),
        "general_replay_retention_within_tolerance": _bool_or_none(
            replay.get("general_replay_retention_within_tolerance")
        ),
        "memory_slot_candidate_slots_scored": _first_int(
            replay.get("brain_installed_memory_slot_candidate_slots_scored"),
            _mapping(training.get("memory_slots")).get("candidate_slots_scored"),
        ),
        "memory_slot_runs_all_slots": _first_bool(
            replay.get("brain_installed_memory_slot_runs_all_slots"),
            _mapping(training.get("memory_slots")).get("runs_all_slots"),
        ),
    }


def _active_compute_evidence(active_compute_category: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _mapping(active_compute_category.get("evidence"))
    return {
        "surface": "marulho_current_language_active_compute_evidence.v1",
        "available": bool(evidence),
        "active_expert_count_per_token": _first_int(
            evidence.get("active_expert_count_per_token")
        ),
        "active_parameters_per_token_estimate": _first_int(
            evidence.get("active_parameters_per_token_estimate")
        ),
        "active_parameter_fraction_estimate": _first_float(
            evidence.get("active_parameter_fraction_estimate")
        ),
        "total_parameters": _first_int(evidence.get("total_parameters")),
        "category_passed": _bool_or_none(active_compute_category.get("passed")),
    }


def _structural_plasticity_evidence(
    growth_prune_category: Mapping[str, Any],
    structural_report: Mapping[str, Any],
) -> dict[str, Any]:
    category_evidence = _mapping(growth_prune_category.get("evidence"))
    saved = _mapping(
        category_evidence.get("brain_installed_structural_plasticity_evidence")
    )
    saved_best = _mapping(saved.get("best_report"))
    summary = _mapping(structural_report.get("structural_transaction_summary"))
    gate = _mapping(structural_report.get("promotion_gate"))
    pre_checkpoint = _mapping(structural_report.get("pre_structural_brain_checkpoint"))
    post_checkpoint = _mapping(structural_report.get("post_structural_brain_checkpoint"))
    sustained = _mapping(structural_report.get("post_structure_sustained_window"))
    best = saved_best or summary or structural_report
    source = (
        "benchmark_suite.brain_installed_structural_plasticity_evidence"
        if saved_best
        else "saved_brain_installed_structural_plasticity_report"
        if structural_report
        else None
    )
    return {
        "surface": "marulho_current_language_structural_plasticity_evidence.v1",
        "available": bool(best),
        "source": source,
        "runtime_owner": _first_string(
            best.get("runtime_owner"),
            structural_report.get("runtime_owner"),
        ),
        "active_language_path": _first_string(
            best.get("active_language_path"),
            structural_report.get("active_language_path"),
        ),
        "brain_surface": _first_string(
            best.get("brain_surface"),
            summary.get("brain_surface"),
        ),
        "training_surface": _first_string(
            best.get("training_surface"),
            summary.get("training_surface"),
        ),
        "trace_event": _first_string(best.get("trace_event"), summary.get("trace_event")),
        "transaction_status": _first_string(
            best.get("transaction_status"),
            summary.get("status"),
            structural_report.get("status"),
        ),
        "proposal_kind": _first_string(best.get("proposal_kind"), summary.get("proposal_kind")),
        "applied": _first_bool(best.get("applied"), summary.get("applied")),
        "operator_approved": _first_bool(
            best.get("operator_approved"),
            summary.get("operator_approved"),
        ),
        "checkpoint_restore_verified": _first_bool(
            best.get("checkpoint_restore_verified"),
            summary.get("checkpoint_restore_verified"),
        ),
        "rollback_verified": _first_bool(
            best.get("rollback_verified"),
            summary.get("rollback_verified"),
            gate.get("records_rollback_evidence"),
        ),
        "heldout_non_regression": _first_bool(
            best.get("heldout_non_regression"),
            summary.get("heldout_non_regression"),
        ),
        "proposal_non_mutating": _first_bool(
            gate.get("proposal_non_mutating"),
            category_evidence.get("route_bank_proposal_mutates_runtime_state") is False
            if "route_bank_proposal_mutates_runtime_state" in category_evidence
            else None,
        ),
        "proposal_runs_through_marulho_brain": _bool_or_none(
            gate.get("proposal_runs_through_marulho_brain")
        ),
        "structural_apply_runs_through_marulho_brain": _bool_or_none(
            gate.get("structural_apply_runs_through_marulho_brain")
        ),
        "records_checkpoint_backed_transaction": _bool_or_none(
            gate.get("records_checkpoint_backed_transaction")
        ),
        "status_read_mutation_absent": _first_bool(
            best.get("status_read_mutation_absent"),
            gate.get("status_read_mutation_absent"),
            structural_report.get("status_read_mutation") is False
            if "status_read_mutation" in structural_report
            else None,
        ),
        "mutation": {
            "surface": "marulho_current_language_structural_mutation_evidence.v1",
            "proposal_kind": _first_string(best.get("proposal_kind"), summary.get("proposal_kind")),
            "source_expert_count": _first_int(
                best.get("source_expert_count"),
                summary.get("source_expert_count"),
            ),
            "target_expert_count": _first_int(
                best.get("target_expert_count"),
                summary.get("target_expert_count"),
            ),
            "source_memory_slot_count": _first_int(
                best.get("source_memory_slot_count"),
                summary.get("source_memory_slot_count"),
            ),
            "target_memory_slot_count": _first_int(
                best.get("target_memory_slot_count"),
                summary.get("target_memory_slot_count"),
            ),
            "memory_slot_count_delta": _first_int(
                best.get("memory_slot_count_delta"),
                summary.get("memory_slot_count_delta"),
            ),
            "source_route_candidate_count": _first_int(
                best.get("source_route_candidate_count"),
                summary.get("source_route_candidate_count"),
                category_evidence.get("route_bank_source_candidate_count"),
            ),
            "target_route_candidate_count": _first_int(
                best.get("target_route_candidate_count"),
                summary.get("target_route_candidate_count"),
                category_evidence.get("route_bank_target_candidate_count"),
            ),
            "route_bank_candidate_count_delta": _first_int(
                best.get("route_bank_candidate_count_delta"),
                summary.get("route_bank_candidate_count_delta"),
            ),
            "route_bank_runs_all_columns": _first_bool(
                category_evidence.get("route_bank_runs_all_columns"),
                False,
            ),
        },
        "pre_structure_checkpoint": {
            "surface": "marulho_current_language_pre_structure_checkpoint_evidence.v1",
            "path": _first_string(
                best.get("pre_structure_checkpoint_path"),
                pre_checkpoint.get("path"),
            ),
            "sha256": _first_string(pre_checkpoint.get("sha256")),
            "restore_verified": _first_bool(
                best.get("pre_structure_checkpoint_restore_verified"),
                pre_checkpoint.get("restore_verified"),
                gate.get("pre_structure_brain_checkpoint_restore_verified"),
            ),
        },
        "post_structure_checkpoint": {
            "surface": "marulho_current_language_post_structure_checkpoint_evidence.v1",
            "path": _first_string(
                best.get("post_structure_checkpoint_path"),
                post_checkpoint.get("path"),
            ),
            "sha256": _first_string(post_checkpoint.get("sha256")),
            "restore_verified": _first_bool(
                best.get("post_structure_checkpoint_restore_verified"),
                post_checkpoint.get("restore_verified"),
                gate.get("post_structure_brain_checkpoint_restore_verified"),
            ),
            "delete_protected_by_current_evidence": bool(
                _first_string(
                    best.get("post_structure_checkpoint_path"),
                    post_checkpoint.get("path"),
                )
            ),
        },
        "post_structure_sustained": {
            "surface": "marulho_current_language_post_structure_sustained_evidence.v1",
            "enabled": _first_bool(
                sustained.get("enabled"),
                best.get("post_structure_sustained_enabled"),
            ),
            "success": _first_bool(
                sustained.get("success"),
                best.get("post_structure_sustained_success"),
            ),
            "target_tokens": _first_int(sustained.get("target_tokens")),
            "token_delta": _first_int(
                sustained.get("token_delta"),
                best.get("post_structure_sustained_token_delta"),
            ),
            "tokens_per_second": _first_float(
                sustained.get("tokens_per_second"),
                best.get("post_structure_sustained_tokens_per_second"),
            ),
            "backend": _first_string(
                sustained.get("backend"),
                best.get("post_structure_sustained_backend"),
            ),
            "device": _first_string(sustained.get("device")),
            "tracked_triton_kernel_failure_count": _first_int(
                sustained.get("tracked_triton_kernel_failure_count"),
                best.get("post_structure_sustained_triton_failure_count"),
            ),
            "tracked_triton_kernel_used_names": _string_list(
                sustained.get("tracked_triton_kernel_used_names")
            ),
        },
        "external_llm_used": _first_bool(structural_report.get("external_llm_used"), False),
        "service_owned_cognition": _first_bool(
            structural_report.get("service_owned_cognition"),
            False,
        ),
        "promotes_runtime_claim": _first_bool(
            best.get("promotes_runtime_claim"),
            structural_report.get("promotes_runtime_claim"),
            gate.get("promotes_runtime_claim"),
            False,
        ),
        "promotes_generation_quality_claim": _first_bool(
            structural_report.get("promotes_generation_quality_claim"),
            gate.get("promotes_generation_quality_claim"),
            False,
        ),
    }


def _checkpoint_lineage_evidence(
    checkpoint_restore_category: Mapping[str, Any],
    rollback_category: Mapping[str, Any],
    training: Mapping[str, Any],
    structural: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_restore = _mapping(checkpoint_restore_category.get("evidence"))
    rollback = _mapping(rollback_category.get("evidence"))
    return {
        "surface": "marulho_current_language_checkpoint_lineage_evidence.v1",
        "available": bool(checkpoint_restore or rollback or training or structural or repair),
        "suite_checkpoint_path": _first_string(checkpoint_restore.get("checkpoint_path")),
        "brain_installed_pre_learning_checkpoint_path": _first_string(
            checkpoint_restore.get("brain_installed_pre_learning_checkpoint_path")
        ),
        "brain_installed_learned_checkpoint_path": _first_string(
            checkpoint_restore.get("brain_installed_learned_checkpoint_path"),
            training.get("learned_brain_checkpoint_path"),
        ),
        "brain_installed_learned_checkpoint_restore_verified": _first_bool(
            checkpoint_restore.get("brain_installed_learned_checkpoint_restore_verified"),
            training.get("learned_brain_checkpoint_restore_verified"),
        ),
        "structural_pre_checkpoint_path": _first_string(
            _mapping(structural.get("pre_structure_checkpoint")).get("path")
        ),
        "structural_pre_checkpoint_restore_verified": _first_bool(
            _mapping(structural.get("pre_structure_checkpoint")).get("restore_verified")
        ),
        "structural_post_checkpoint_path": _first_string(
            _mapping(structural.get("post_structure_checkpoint")).get("path")
        ),
        "structural_post_checkpoint_restore_verified": _first_bool(
            _mapping(structural.get("post_structure_checkpoint")).get("restore_verified")
        ),
        "selected_repair_checkpoint_path": _first_string(
            repair.get("selected_checkpoint_path")
        ),
        "selected_repair_checkpoint_restore_verified": _first_bool(
            repair.get("checkpoint_restore_verified")
        ),
        "checkpoint_evolution": {
            "surface": "marulho_current_language_checkpoint_evolution_evidence.v1",
            "checkpoint_lineage_complete": _bool_or_none(
                rollback.get("checkpoint_lineage_complete")
            ),
            "lineage_id": _first_string(rollback.get("lineage_id")),
            "parent_kept_installed": _bool_or_none(rollback.get("parent_kept_installed")),
            "parent_runtime_unchanged": _bool_or_none(
                rollback.get("parent_runtime_unchanged")
            ),
            "rollback_to_parent_verified": _bool_or_none(
                rollback.get("rollback_to_parent_verified")
            ),
            "operator_review_required": _bool_or_none(
                rollback.get("operator_review_required")
            ),
            "long_run_evidence_required_for_promotion": _bool_or_none(
                rollback.get("long_run_evidence_required_for_promotion")
            ),
            "saved_checkpoint_evolution_available": _bool_or_none(
                _mapping(rollback.get("saved_checkpoint_evolution_evidence")).get(
                    "checkpoint_evolution_evidence_available"
                )
            ),
            "promotes_runtime_claim": _first_bool(
                _mapping(rollback.get("saved_checkpoint_evolution_evidence")).get(
                    "promotes_runtime_claim"
                ),
                False,
            ),
        },
    }


def _backend_bottleneck_evidence(
    training: Mapping[str, Any],
    state_block_impact: Mapping[str, Any],
    eligibility_impact: Mapping[str, Any],
    memory_slot_training_impact: Mapping[str, Any],
    latest_memory_slot_training_impact: Mapping[str, Any],
) -> dict[str, Any]:
    state_cmp = _mapping(state_block_impact.get("comparison"))
    eligibility_cmp = _mapping(eligibility_impact.get("comparison"))
    memory_cmp = _mapping(memory_slot_training_impact.get("comparison"))
    memory_review = _mapping(memory_slot_training_impact.get("review"))
    latest_memory_report = _mapping(latest_memory_slot_training_impact)
    memory_slots = _mapping(training.get("memory_slots"))
    decisions = []
    if state_cmp:
        decisions.append(
            {
                "surface": "marulho_current_language_backend_decision.v1",
                "name": "state_block_preallocation",
                "status": "rejected_as_default",
                "reason": "complete_forward_ratio_below_current_stacked_path",
                "baseline_tokens_per_second": _first_float(
                    state_cmp.get("baseline_tokens_per_second")
                ),
                "candidate_tokens_per_second": _first_float(
                    state_cmp.get("preallocated_tokens_per_second")
                ),
                "candidate_vs_baseline_ratio": _first_float(
                    state_cmp.get("preallocated_vs_baseline_tokens_per_second_ratio")
                ),
                "parity_passed": _bool_or_none(state_cmp.get("parity_passed")),
            }
        )
    if eligibility_cmp:
        decisions.append(
            {
                "surface": "marulho_current_language_backend_decision.v1",
                "name": "deferred_eligibility_trace_scan",
                "status": "rejected_as_default",
                "reason": "complete_forward_ratio_below_inline_plif_update",
                "baseline_tokens_per_second": _first_float(
                    eligibility_cmp.get("baseline_tokens_per_second")
                ),
                "candidate_tokens_per_second": _first_float(
                    eligibility_cmp.get("deferred_tokens_per_second")
                ),
                "candidate_vs_baseline_ratio": _first_float(
                    eligibility_cmp.get("deferred_vs_baseline_tokens_per_second_ratio")
                ),
                "parity_passed": _bool_or_none(eligibility_cmp.get("parity_passed")),
            }
        )
    if memory_cmp:
        decisions.append(
            {
                "surface": "marulho_current_language_backend_decision.v1",
                "name": "memory_slot_triton_training_autograd",
                "status": "opt_in_rejection_for_current_default",
                "reason": (
                    "isolated optimizer-step Triton can beat bounded torch, but "
                    "current installed-brain complete-window evidence still uses "
                    "the measured torch-autograd bounded memory-slot backend"
                ),
                "accepted_current_backend": _first_string(
                    memory_slots.get("retrieval_backend")
                ),
                "control_tokens_per_second": _first_float(
                    memory_cmp.get("control_tokens_per_second")
                ),
                "bounded_tokens_per_second": _first_float(
                    memory_cmp.get("bounded_tokens_per_second")
                ),
                "triton_training_tokens_per_second": _first_float(
                    memory_cmp.get("triton_training_tokens_per_second")
                ),
                "triton_training_vs_bounded_ratio": _first_float(
                    memory_cmp.get("triton_training_vs_bounded_tokens_per_second_ratio")
                ),
                "triton_training_vs_control_ratio": _first_float(
                    memory_cmp.get("triton_training_vs_control_tokens_per_second_ratio")
                ),
                "bounded_avoids_all_slot_scan": _bool_or_none(
                    memory_cmp.get("bounded_avoids_all_slot_scan")
                ),
                "report_status": _string_or_none(
                    memory_slot_training_impact.get("report_status")
                ),
                "hot_update_evidence_mode": _string_or_none(
                    memory_review.get("hot_update_evidence_mode")
                ),
                "per_step_evidence_dict_build": _bool_or_none(
                    memory_review.get("per_step_evidence_dict_build")
                ),
                "per_step_memory_slot_stats_delta": _bool_or_none(
                    memory_review.get("per_step_memory_slot_stats_delta")
                ),
            }
        )
    return {
        "surface": "marulho_current_language_backend_bottleneck_evidence.v1",
        "read_only_projection": True,
        "current_training_backend": _first_string(memory_slots.get("retrieval_backend")),
        "current_training_update_tokens_per_second": _first_float(
            training.get("tokens_per_second")
        ),
        "current_training_total_window_tokens_per_second": _first_float(
            training.get("total_window_tokens_per_second")
        ),
        "complete_window_evidence_required_for_default_change": True,
        "memory_slot_training_report_selection": {
            "surface": (
                "marulho_current_language_memory_slot_training_report_selection.v1"
            ),
            "latest_report_status": _string_or_none(
                latest_memory_report.get("report_status")
            ),
            "latest_report_complete": (
                _report_is_complete(latest_memory_report)
                if latest_memory_report
                else None
            ),
            "latest_completed_arm_names": _string_list(
                latest_memory_report.get("completed_arm_names")
            ),
            "latest_missing_arm_names": _string_list(
                latest_memory_report.get("missing_arm_names")
            ),
            "latest_partial_reason": _string_or_none(
                latest_memory_report.get("partial_reason")
            ),
            "backend_decision_report_status": _string_or_none(
                memory_slot_training_impact.get("report_status")
            ),
            "backend_decision_uses_latest_report": bool(
                latest_memory_report
                and latest_memory_slot_training_impact is memory_slot_training_impact
            ),
            "partial_reports_do_not_replace_complete_backend_decision": True,
        },
        "decision_count": len(decisions),
        "decisions": decisions,
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
