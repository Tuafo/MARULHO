"""Build a same-child quality-retention review bundle for MARULHO LM reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_quality_retention_review_bundle.v1"
ARTIFACT_KIND = "marulho_language_quality_retention_review_bundle"
QUALITY_REPLAY_SURFACE = "marulho_language_quality_replay_experiment.v1"
BENCHMARK_SUITE_SURFACE = "marulho_language_runtime_benchmark_suite.v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _path_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _resolve_artifact_path(value: Any, *, base_dir: str | Path | None = None) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return Path(base_dir or Path.cwd()) / path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _selected_candidate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = _mapping(report.get("candidate_selection"))
    selected_id = str(selection.get("selected_candidate_id") or "")
    for candidate in _list(selection.get("candidates")):
        item = _mapping(candidate)
        if item.get("selected") is True:
            return item
    for candidate in _list(selection.get("candidates")):
        item = _mapping(candidate)
        if selected_id and str(item.get("candidate_id") or "") == selected_id:
            return item
    return {}


def _selected_child_checkpoint(report: Mapping[str, Any]) -> dict[str, str]:
    selection = _mapping(report.get("candidate_selection"))
    return {
        "path": str(
            selection.get("selected_child_checkpoint_path")
            or report.get("child_checkpoint_path")
            or ""
        ),
        "sha256": str(
            selection.get("selected_child_checkpoint_sha256")
            or report.get("child_checkpoint_sha256")
            or ""
        ),
    }


def _case_loss(case: Mapping[str, Any]) -> float | None:
    loss = case.get("source_continuation_loss")
    if isinstance(loss, Mapping):
        loss = loss.get("loss")
    try:
        return float(loss) if loss is not None else None
    except (TypeError, ValueError):
        return None


def _case_perplexity(case: Mapping[str, Any]) -> float | None:
    perplexity = case.get("source_continuation_perplexity")
    if perplexity is None and isinstance(case.get("source_continuation_loss"), Mapping):
        perplexity = _mapping(case.get("source_continuation_loss")).get("perplexity")
    try:
        return float(perplexity) if perplexity is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _raw_continuation_flags(case: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if case.get("passed") is False:
        flags.append("case_failed")
    for reason in _list(case.get("failure_reasons")):
        if reason:
            flags.append(str(reason))
    continuation = str(
        case.get("continuation_text")
        or case.get("generated_continuation")
        or case.get("generated_text")
        or ""
    )
    if "\ufffd" in continuation:
        flags.append("replacement_character_present")
    prefix_fraction = _float_or_none(case.get("prefix_match_fraction"))
    if prefix_fraction is not None and prefix_fraction < 0.75:
        flags.append("low_prefix_match_fraction")
    printable_fraction = _float_or_none(case.get("printable_fraction"))
    if printable_fraction is not None and printable_fraction < 0.98:
        flags.append("low_printable_fraction")
    distinct_bigram = _float_or_none(case.get("distinct_bigram_fraction"))
    if distinct_bigram is not None and distinct_bigram < 0.15:
        flags.append("low_distinct_bigram_fraction")
    token_run = _int_or_zero(case.get("max_token_run_length"))
    if token_run > 8:
        flags.append("long_token_run")
    return sorted(set(flags))


def _raw_continuation_samples(
    report: Mapping[str, Any],
    *,
    phase: str,
    limit: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case in _list(report.get("cases"))[: max(0, limit)]:
        item = _mapping(case)
        samples.append(
            {
                "phase": phase,
                "prompt_text": str(item.get("prompt_text") or item.get("prompt") or ""),
                "expected_source_continuation": str(
                    item.get("expected_source_continuation")
                    or item.get("expected_continuation")
                    or ""
                ),
                "continuation_text": str(
                    item.get("continuation_text")
                    or item.get("generated_continuation")
                    or item.get("generated_text")
                    or ""
                ),
                "passed": bool(item.get("passed")),
                "failure_reasons": [
                    str(reason) for reason in _list(item.get("failure_reasons"))
                ],
                "prefix_match_chars": item.get("prefix_match_chars"),
                "prefix_match_fraction": item.get("prefix_match_fraction"),
                "source_continuation_loss": _case_loss(item),
                "source_continuation_perplexity": _case_perplexity(item),
                "automated_raw_text_flags": _raw_continuation_flags(item),
            }
        )
    return samples


def _coherence_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    gate = _mapping(report.get("promotion_gate"))
    return {
        "available": bool(report),
        "checkpoint_path": str(report.get("checkpoint_path") or ""),
        "generation_coherence_available": bool(
            gate.get("generation_coherence_available")
            if "generation_coherence_available" in gate
            else report
        ),
        "case_count": _int_or_zero(summary.get("case_count")),
        "passed_case_count": _int_or_zero(summary.get("passed_case_count")),
        "case_pass_rate": _float_or_none(summary.get("case_pass_rate")),
        "mean_prefix_match_chars": _float_or_none(
            summary.get("mean_prefix_match_chars")
        ),
        "mean_prefix_match_fraction": _float_or_none(
            summary.get("mean_prefix_match_fraction")
        ),
        "mean_source_continuation_loss": _float_or_none(
            summary.get("mean_source_continuation_loss")
        ),
        "mean_source_continuation_perplexity": _float_or_none(
            summary.get("mean_source_continuation_perplexity")
        ),
        "source_continuation_loss_available": bool(
            summary.get("source_continuation_loss_available")
        ),
        "raw_case_count": len(_list(report.get("cases"))),
    }


def _coherence_block(
    *,
    name: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    delta: Mapping[str, Any],
    expected_child_checkpoint_path: str,
    raw_sample_limit: int,
) -> dict[str, Any]:
    before_samples = _raw_continuation_samples(
        before,
        phase="before",
        limit=raw_sample_limit,
    )
    after_samples = _raw_continuation_samples(
        after,
        phase="after",
        limit=raw_sample_limit,
    )
    all_samples = before_samples + after_samples
    flagged_samples = [
        sample for sample in all_samples if sample["automated_raw_text_flags"]
    ]
    after_checkpoint = str(after.get("checkpoint_path") or "")
    return {
        "surface": "marulho_language_quality_retention_prompt_review.v1",
        "name": name,
        "before": _coherence_summary(before),
        "after": _coherence_summary(after),
        "delta": {
            "source_continuation_loss_available": bool(
                delta.get("source_continuation_loss_available")
            ),
            "passed_case_count_delta": _int_or_zero(
                delta.get("passed_case_count_delta")
            ),
            "case_pass_rate_delta": _float_or_none(
                delta.get("case_pass_rate_delta")
            ),
            "mean_source_continuation_loss_delta": _float_or_none(
                delta.get("mean_source_continuation_loss_delta")
            ),
            "mean_source_continuation_loss_regressed": bool(
                delta.get("mean_source_continuation_loss_regressed")
            ),
            "mean_source_continuation_perplexity_delta": _float_or_none(
                delta.get("mean_source_continuation_perplexity_delta")
            ),
            "mean_source_continuation_perplexity_regressed": bool(
                delta.get("mean_source_continuation_perplexity_regressed")
            ),
            "prompt_pass_nonregressed": bool(
                delta.get("prompt_pass_nonregressed")
            ),
            "prompt_pass_nonregressed_but_loss_regressed": bool(
                delta.get("prompt_pass_nonregressed_but_loss_regressed")
            ),
            "repaired_prompt_count": _int_or_zero(delta.get("repaired_prompt_count")),
            "repaired_prompts": [str(item) for item in _list(delta.get("repaired_prompts"))],
            "regressed_prompt_count": _int_or_zero(
                delta.get("regressed_prompt_count")
            ),
            "regressed_prompts": [
                str(item) for item in _list(delta.get("regressed_prompts"))
            ],
            "promotes_generation_quality_claim": False,
        },
        "same_child_checkpoint": bool(
            expected_child_checkpoint_path
            and _path_key(after_checkpoint) == _path_key(expected_child_checkpoint_path)
        ),
        "expected_child_checkpoint_path": expected_child_checkpoint_path,
        "after_checkpoint_path": after_checkpoint,
        "raw_continuation_review": {
            "surface": "marulho_language_quality_retention_raw_continuation_review.v1",
            "sample_limit_per_phase": int(raw_sample_limit),
            "sample_count": len(all_samples),
            "flagged_sample_count": len(flagged_samples),
            "flag_reasons": sorted(
                {
                    reason
                    for sample in flagged_samples
                    for reason in sample["automated_raw_text_flags"]
                }
            ),
            "samples": all_samples,
            "requires_human_raw_continuation_review": True,
            "human_review_satisfied": False,
            "promotes_generation_quality_claim": False,
        },
    }


def _learning_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    learning_report = _mapping(report.get("learning_evidence"))
    evidence = _mapping(learning_report.get("learning_evidence"))
    active_compute = _mapping(
        evidence.get("active_compute") or learning_report.get("active_compute")
    )
    rollback = _mapping(learning_report.get("rollback_evidence"))
    gate = _mapping(learning_report.get("promotion_gate"))
    return {
        "surface": "marulho_language_quality_retention_learning_summary.v1",
        "status": learning_report.get("status"),
        "eligible_for_online_learning_review": bool(
            gate.get("eligible_for_online_learning_review")
        ),
        "rollback_available": bool(gate.get("rollback_available")),
        "rollback_restore_verified": bool(rollback.get("restore_verified")),
        "update_token_count": evidence.get("update_token_count"),
        "tokens_per_second": evidence.get("tokens_per_second"),
        "total_window_tokens_per_second": evidence.get(
            "total_window_tokens_per_second"
        ),
        "new_domain_loss_delta": evidence.get("new_domain_loss_delta"),
        "old_domain_forgetting": evidence.get("old_domain_forgetting"),
        "general_replay_retention_delta": evidence.get(
            "general_replay_retention_delta"
        ),
        "active_compute_recorded": bool(active_compute),
        "active_compute": dict(active_compute),
        "external_llm_used": learning_report.get("external_llm_used"),
        "loads_external_checkpoint": learning_report.get("loads_external_checkpoint"),
        "active_language_path": learning_report.get("active_language_path"),
    }


def _sustained_summary(
    report: Mapping[str, Any],
    *,
    expected_child_checkpoint_path: str,
) -> dict[str, Any]:
    summary = _mapping(report.get("sustained_runtime_evidence_summary"))
    reports = [
        _mapping(item)
        for item in _list(
            report.get("sustained_runtime_evidence_reports") or summary.get("reports")
        )
    ]
    matching = [
        item
        for item in reports
        if _path_key(item.get("checkpoint_path"))
        == _path_key(expected_child_checkpoint_path)
    ]
    house_matching = [
        item
        for item in matching
        if _int_or_zero(item.get("token_delta") or item.get("target_tokens")) >= 524288
    ]
    optional_8388608 = [
        item
        for item in matching
        if _int_or_zero(item.get("token_delta") or item.get("target_tokens"))
        >= 8388608
    ]
    return {
        "surface": "marulho_language_quality_retention_sustained_summary.v1",
        "enabled": bool(summary.get("enabled")),
        "all_success": bool(summary.get("all_success")),
        "controlled_decode_all_success": bool(
            summary.get("controlled_decode_all_success")
        ),
        "house_scale_524288_available": bool(
            summary.get("house_scale_524288_available")
        ),
        "controlled_decode_house_scale_524288_available": bool(
            summary.get("controlled_decode_house_scale_524288_available")
        ),
        "target_tokens": list(summary.get("target_tokens") or []),
        "min_tokens_per_second": summary.get("min_tokens_per_second"),
        "max_tokens_per_second": summary.get("max_tokens_per_second"),
        "report_count": _int_or_zero(summary.get("report_count")),
        "same_child_report_count": len(matching),
        "same_child_house_scale_524288_report_count": len(house_matching),
        "same_child_optional_8388608_report_count": len(optional_8388608),
        "same_child_reports": [dict(item) for item in matching],
    }


def _candidate_retention_review(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(report.get("candidate_selection"))
    review = _mapping(selection.get("selected_quality_retention_review"))
    if not review:
        review = _mapping(_selected_candidate(report).get("quality_retention_review"))
    return {
        "surface": "marulho_language_quality_retention_selected_candidate_review.v1",
        "available": bool(review),
        "suspicious": bool(review.get("suspicious")),
        "suspicious_reasons": [
            str(item) for item in _list(review.get("suspicious_reasons"))
        ],
        "trained_prompt_pass_nonregressed_but_loss_regressed": bool(
            review.get("trained_prompt_pass_nonregressed_but_loss_regressed")
        ),
        "heldout_prompt_pass_nonregressed_but_loss_regressed": bool(
            review.get("heldout_prompt_pass_nonregressed_but_loss_regressed")
        ),
        "trained_source_continuation_loss_delta": review.get(
            "trained_source_continuation_loss_delta"
        ),
        "heldout_source_continuation_loss_delta": review.get(
            "heldout_source_continuation_loss_delta"
        ),
        "promotes_generation_quality_claim": False,
    }


def _overlap_review(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(report.get("candidate_selection"))
    heldout_suite = _mapping(report.get("heldout_prompt_suite"))
    fresh_suite = _mapping(report.get("fresh_heldout_prompt_suite"))
    return {
        "surface": "marulho_language_quality_retention_prompt_overlap_review.v1",
        "heldout_not_used_for_replay_training": bool(
            heldout_suite.get("not_used_for_replay_training")
        )
        and selection.get("heldout_cases_used_for_replay_training") is False,
        "heldout_training_prompt_overlap_count": _int_or_zero(
            selection.get("heldout_training_prompt_overlap_count")
            or heldout_suite.get("training_prompt_overlap_count")
        ),
        "heldout_training_prompt_overlaps": [
            str(item)
            for item in _list(
                selection.get("heldout_training_prompt_overlaps")
                or heldout_suite.get("training_prompt_overlaps")
            )
        ],
        "fresh_heldout_enabled": bool(fresh_suite.get("enabled")),
        "fresh_heldout_built_after_candidate_selection": bool(
            fresh_suite.get("built_after_candidate_selection")
        ),
        "fresh_heldout_not_used_for_replay_training": bool(
            fresh_suite.get("not_used_for_replay_training")
        )
        and not bool(selection.get("fresh_heldout_cases_used_for_replay_training")),
        "fresh_heldout_training_prompt_overlap_count": _int_or_zero(
            selection.get("fresh_heldout_training_prompt_overlap_count")
            or fresh_suite.get("training_prompt_overlap_count")
        ),
        "fresh_heldout_fixed_prompt_overlap_count": _int_or_zero(
            selection.get("fresh_heldout_fixed_prompt_overlap_count")
            or fresh_suite.get("fixed_heldout_prompt_overlap_count")
        ),
    }


def _candidate_mode(candidate: Mapping[str, Any]) -> str:
    learning_config = _mapping(candidate.get("learning_config"))
    return str(learning_config.get("replay_gradient_projection_mode") or "")


def _candidate_delta_loss(candidate: Mapping[str, Any], key: str) -> float | None:
    return _float_or_none(_mapping(candidate.get(key)).get("mean_source_continuation_loss_delta"))


def _candidate_regressed_count(candidate: Mapping[str, Any], key: str) -> int:
    return _int_or_zero(_mapping(candidate.get(key)).get("regressed_prompt_count"))


def _projection_ablation_summary(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    promoted = False
    rejection_reasons: list[str] = []
    for report in reports:
        selection = _mapping(report.get("candidate_selection"))
        candidates = [_mapping(item) for item in _list(selection.get("candidates"))]
        disabled = next(
            (candidate for candidate in candidates if _candidate_mode(candidate) == "disabled"),
            {},
        )
        dense = next(
            (candidate for candidate in candidates if _candidate_mode(candidate) == "dense_core"),
            {},
        )
        selected = _selected_candidate(report)
        dense_quality_not_worse = False
        dense_throughput_not_worse = False
        if disabled and dense:
            dense_quality_not_worse = all(
                [
                    (_candidate_delta_loss(dense, "trained_generation_coherence_delta") or 0.0)
                    <= (
                        _candidate_delta_loss(
                            disabled,
                            "trained_generation_coherence_delta",
                        )
                        or 0.0
                    ),
                    (_candidate_delta_loss(dense, "heldout_generation_coherence_delta") or 0.0)
                    <= (
                        _candidate_delta_loss(
                            disabled,
                            "heldout_generation_coherence_delta",
                        )
                        or 0.0
                    ),
                    _candidate_regressed_count(
                        dense,
                        "trained_generation_coherence_delta",
                    )
                    <= _candidate_regressed_count(
                        disabled,
                        "trained_generation_coherence_delta",
                    ),
                    _candidate_regressed_count(
                        dense,
                        "heldout_generation_coherence_delta",
                    )
                    <= _candidate_regressed_count(
                        disabled,
                        "heldout_generation_coherence_delta",
                    ),
                ]
            )
            dense_throughput_not_worse = float(
                dense.get("total_window_tokens_per_second", 0.0) or 0.0
            ) >= float(disabled.get("total_window_tokens_per_second", 0.0) or 0.0)
            if not dense_quality_not_worse:
                rejection_reasons.append("dense_core_projection_quality_not_equal_or_better")
            if not dense_throughput_not_worse:
                rejection_reasons.append("dense_core_projection_total_window_slower")
        selected_review = _mapping(selection.get("selected_quality_retention_review"))
        if selected_review.get("suspicious") is True:
            rejection_reasons.append("selected_projection_candidate_suspicious")
        entries.append(
            {
                "path": str(report.get("path") or report.get("output_path") or ""),
                "selected_candidate_id": selection.get("selected_candidate_id"),
                "selected_projection_mode": _candidate_mode(selected),
                "disabled_candidate": _projection_candidate_summary(disabled),
                "dense_core_candidate": _projection_candidate_summary(dense),
                "dense_core_quality_equal_or_better": dense_quality_not_worse,
                "dense_core_total_window_throughput_equal_or_better": (
                    dense_throughput_not_worse
                ),
            }
        )
    promoted = bool(entries) and not rejection_reasons
    return {
        "surface": "marulho_language_quality_retention_projection_ablation_summary.v1",
        "report_count": len(entries),
        "entries": entries,
        "projection_promoted": promoted,
        "projection_rejection_reasons": sorted(set(rejection_reasons)),
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }


def _projection_candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "projection_mode": _candidate_mode(candidate),
        "selected": bool(candidate.get("selected")),
        "update_tokens_per_second": candidate.get("update_tokens_per_second"),
        "total_window_tokens_per_second": candidate.get(
            "total_window_tokens_per_second"
        ),
        "trained_loss_delta": _candidate_delta_loss(
            candidate,
            "trained_generation_coherence_delta",
        ),
        "heldout_loss_delta": _candidate_delta_loss(
            candidate,
            "heldout_generation_coherence_delta",
        ),
        "trained_regressed_prompt_count": _candidate_regressed_count(
            candidate,
            "trained_generation_coherence_delta",
        ),
        "heldout_regressed_prompt_count": _candidate_regressed_count(
            candidate,
            "heldout_generation_coherence_delta",
        ),
        "quality_retention_suspicious": bool(
            _mapping(candidate.get("quality_retention_review")).get("suspicious")
        ),
    }


def _benchmark_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    if not report:
        return {
            "available": False,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
        }
    gate = _mapping(report.get("promotion_gate"))
    return {
        "surface": "marulho_language_quality_retention_benchmark_suite_summary.v1",
        "available": True,
        "suite_surface_valid": report.get("surface") == BENCHMARK_SUITE_SURFACE,
        "status": gate.get("status"),
        "failed_category_names": [
            str(item) for item in _list(gate.get("failed_category_names"))
        ],
        "missing_required_category_names": [
            str(item) for item in _list(gate.get("missing_required_category_names"))
        ],
        "quality_replay_evidence_available": bool(
            gate.get("quality_replay_evidence_available")
        ),
        "quality_replay_controlled_decode_house_scale_aligned": bool(
            gate.get("quality_replay_controlled_decode_house_scale_aligned")
        ),
        "promotes_runtime_claim": bool(gate.get("promotes_runtime_claim")),
        "promotes_generation_quality_claim": False,
    }


def _required_and_failed_evidence(
    *,
    quality: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_file: Mapping[str, Any],
    trained: Mapping[str, Any],
    heldout: Mapping[str, Any],
    fresh: Mapping[str, Any],
    learning: Mapping[str, Any],
    sustained: Mapping[str, Any],
    candidate_review: Mapping[str, Any],
    overlap_review: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str], dict[str, bool], list[str]]:
    trained_delta = _mapping(trained.get("delta"))
    heldout_delta = _mapping(heldout.get("delta"))
    fresh_delta = _mapping(fresh.get("delta"))
    required = {
        "quality_replay_surface_valid": quality.get("surface") == QUALITY_REPLAY_SURFACE,
        "marulho_owned_language_path": quality.get("owned_by_marulho") is True
        and quality.get("active_language_path") == "marulho_lm_head",
        "external_llm_absent": quality.get("external_llm_used") is False,
        "external_checkpoint_loader_absent": quality.get("loads_external_checkpoint")
        is False,
        "selected_child_checkpoint_path_available": bool(checkpoint.get("path")),
        "selected_child_checkpoint_sha256_available": bool(checkpoint.get("sha256")),
        "selected_child_checkpoint_file_exists": bool(checkpoint_file.get("exists")),
        "selected_child_checkpoint_hash_verified": bool(
            checkpoint_file.get("hash_verified")
        ),
        "trained_prompt_after_available": bool(_mapping(trained.get("after")).get("available")),
        "trained_prompt_after_same_child": bool(trained.get("same_child_checkpoint")),
        "trained_source_continuation_loss_available": bool(
            trained_delta.get("source_continuation_loss_available")
        ),
        "heldout_prompt_after_available": bool(_mapping(heldout.get("after")).get("available")),
        "heldout_prompt_after_same_child": bool(heldout.get("same_child_checkpoint")),
        "heldout_source_continuation_loss_available": bool(
            heldout_delta.get("source_continuation_loss_available")
        ),
        "fresh_heldout_prompt_after_available": bool(
            _mapping(fresh.get("after")).get("available")
        ),
        "fresh_heldout_prompt_after_same_child": bool(
            fresh.get("same_child_checkpoint")
        ),
        "fresh_heldout_source_continuation_loss_available": bool(
            fresh_delta.get("source_continuation_loss_available")
        ),
        "selected_candidate_quality_retention_review_available": bool(
            candidate_review.get("available")
        ),
        "heldout_no_training_prompt_overlap": bool(
            overlap_review.get("heldout_not_used_for_replay_training")
        )
        and _int_or_zero(overlap_review.get("heldout_training_prompt_overlap_count"))
        == 0,
        "fresh_heldout_built_after_candidate_selection": bool(
            overlap_review.get("fresh_heldout_enabled")
        )
        and bool(overlap_review.get("fresh_heldout_built_after_candidate_selection")),
        "fresh_heldout_no_training_prompt_overlap": bool(
            overlap_review.get("fresh_heldout_not_used_for_replay_training")
        )
        and _int_or_zero(
            overlap_review.get("fresh_heldout_training_prompt_overlap_count")
        )
        == 0,
        "fresh_heldout_no_fixed_heldout_prompt_overlap": _int_or_zero(
            overlap_review.get("fresh_heldout_fixed_prompt_overlap_count")
        )
        == 0,
        "learning_update_accepted": learning.get("status")
        == "accepted_online_update",
        "learning_rollback_verified": bool(learning.get("rollback_restore_verified")),
        "learning_old_new_replay_losses_recorded": all(
            key in learning
            and learning.get(key) is not None
            for key in (
                "new_domain_loss_delta",
                "old_domain_forgetting",
                "general_replay_retention_delta",
            )
        ),
        "active_compute_recorded": bool(learning.get("active_compute_recorded")),
        "same_child_524288_sustained_decode_available": bool(
            sustained.get("house_scale_524288_available")
        )
        and _int_or_zero(
            sustained.get("same_child_house_scale_524288_report_count")
        )
        > 0,
        "same_child_controlled_decode_524288_available": bool(
            sustained.get("controlled_decode_house_scale_524288_available")
        ),
        "same_child_sustained_decode_success": bool(sustained.get("all_success"))
        and bool(sustained.get("controlled_decode_all_success")),
        "raw_continuation_samples_available": all(
            _mapping(item.get("raw_continuation_review")).get("sample_count", 0)
            for item in (trained, heldout, fresh)
        ),
    }
    failed = {
        "trained_prompt_regression_absent": _int_or_zero(
            trained_delta.get("regressed_prompt_count")
        )
        == 0,
        "heldout_prompt_regression_absent": _int_or_zero(
            heldout_delta.get("regressed_prompt_count")
        )
        == 0,
        "fresh_heldout_prompt_regression_absent": _int_or_zero(
            fresh_delta.get("regressed_prompt_count")
        )
        == 0,
        "trained_pass_without_loss_regression": not bool(
            trained_delta.get("prompt_pass_nonregressed_but_loss_regressed")
        ),
        "heldout_pass_without_loss_regression": not bool(
            heldout_delta.get("prompt_pass_nonregressed_but_loss_regressed")
        ),
        "fresh_heldout_pass_without_loss_regression": not bool(
            fresh_delta.get("prompt_pass_nonregressed_but_loss_regressed")
        ),
        "selected_candidate_not_suspicious": not bool(
            candidate_review.get("suspicious")
        ),
    }
    missing = [name for name, passed in required.items() if not bool(passed)]
    failed_names = [name for name, passed in failed.items() if not bool(passed)]
    return required, missing, failed, failed_names


def build_language_quality_retention_review_bundle(
    *,
    output_path: str | Path,
    quality_replay_evidence_path: str | Path,
    benchmark_suite_evidence_path: str | Path | None = None,
    projection_ablation_evidence_paths: Sequence[str | Path] = (),
    base_dir: str | Path | None = None,
    raw_sample_limit: int = 12,
) -> dict[str, Any]:
    """Aggregate quality-retention evidence without training or mutation."""

    output = Path(output_path)
    quality_path = Path(quality_replay_evidence_path)
    quality = _read_json(quality_path)
    checkpoint = _selected_child_checkpoint(quality)
    selected_path = _resolve_artifact_path(checkpoint["path"], base_dir=base_dir)
    checkpoint_exists = selected_path.is_file()
    checkpoint_file_sha = _sha256_file(selected_path) if checkpoint_exists else ""
    checkpoint_file = {
        "exists": checkpoint_exists,
        "resolved_path": str(selected_path),
        "sha256": checkpoint_file_sha,
        "hash_verified": bool(
            checkpoint.get("sha256") and checkpoint_file_sha == checkpoint.get("sha256")
        ),
    }

    expected_child_path = str(checkpoint.get("path") or "")
    trained = _coherence_block(
        name="trained_prompt_bank",
        before=_mapping(quality.get("generation_coherence_before")),
        after=_mapping(quality.get("generation_coherence_after")),
        delta=_mapping(quality.get("generation_coherence_delta")),
        expected_child_checkpoint_path=expected_child_path,
        raw_sample_limit=raw_sample_limit,
    )
    heldout = _coherence_block(
        name="fixed_heldout_prompt_bank",
        before=_mapping(quality.get("heldout_generation_coherence_before")),
        after=_mapping(quality.get("heldout_generation_coherence_after")),
        delta=_mapping(quality.get("heldout_generation_coherence_delta")),
        expected_child_checkpoint_path=expected_child_path,
        raw_sample_limit=raw_sample_limit,
    )
    fresh = _coherence_block(
        name="fresh_post_selection_heldout_prompt_bank",
        before=_mapping(quality.get("fresh_heldout_generation_coherence_before")),
        after=_mapping(quality.get("fresh_heldout_generation_coherence_after")),
        delta=_mapping(quality.get("fresh_heldout_generation_coherence_delta")),
        expected_child_checkpoint_path=expected_child_path,
        raw_sample_limit=raw_sample_limit,
    )
    learning = _learning_summary(quality)
    sustained = _sustained_summary(
        quality,
        expected_child_checkpoint_path=expected_child_path,
    )
    candidate_review = _candidate_retention_review(quality)
    overlap_review = _overlap_review(quality)
    required, missing, failed, failed_names = _required_and_failed_evidence(
        quality=quality,
        checkpoint=checkpoint,
        checkpoint_file=checkpoint_file,
        trained=trained,
        heldout=heldout,
        fresh=fresh,
        learning=learning,
        sustained=sustained,
        candidate_review=candidate_review,
        overlap_review=overlap_review,
    )

    benchmark_report = (
        _read_json(benchmark_suite_evidence_path)
        if benchmark_suite_evidence_path is not None
        else {}
    )
    projection_reports = [
        _read_json(path) for path in projection_ablation_evidence_paths
    ]
    status = (
        "blocked_missing_required_evidence"
        if missing
        else "blocked_quality_retention"
        if failed_names
        else "ready_for_review"
    )
    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ready_for_review": status == "ready_for_review",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": str(
            quality.get("active_language_path") or "marulho_lm_head"
        ),
        "output_path": str(output),
        "quality_replay_evidence_path": str(quality_path),
        "benchmark_suite_evidence_path": (
            None
            if benchmark_suite_evidence_path is None
            else str(Path(benchmark_suite_evidence_path))
        ),
        "projection_ablation_evidence_paths": [
            str(Path(path)) for path in projection_ablation_evidence_paths
        ],
        "selected_child_checkpoint": {
            "surface": "marulho_language_quality_retention_selected_child_checkpoint.v1",
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "file_exists": checkpoint_file["exists"],
            "resolved_path": checkpoint_file["resolved_path"],
            "file_sha256": checkpoint_file["sha256"],
            "hash_verified": checkpoint_file["hash_verified"],
        },
        "checkpoint_lineage": dict(_mapping(quality.get("checkpoint_lineage"))),
        "prompt_retention": {
            "surface": "marulho_language_quality_retention_prompt_banks.v1",
            "trained_prompt_bank": trained,
            "fixed_heldout_prompt_bank": heldout,
            "fresh_post_selection_heldout_prompt_bank": fresh,
        },
        "learning_retention": learning,
        "active_compute": dict(_mapping(learning.get("active_compute"))),
        "sustained_decode": sustained,
        "candidate_quality_retention_review": candidate_review,
        "prompt_overlap_review": overlap_review,
        "quality_generalization_review": dict(
            _mapping(quality.get("quality_generalization_review"))
        ),
        "benchmark_suite_summary": _benchmark_summary(benchmark_report),
        "projection_ablation_summary": _projection_ablation_summary(
            projection_reports
        ),
        "review_gate": {
            "surface": "marulho_language_quality_retention_review_gate.v1",
            "status": status,
            "required_evidence": required,
            "missing_evidence": missing,
            "quality_checks": failed,
            "failed_quality_checks": failed_names,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
            "next_gate": (
                "operator_raw_continuation_and_quality_review"
                if status == "ready_for_review"
                else "collect_missing_quality_retention_evidence"
                if missing
                else "repair_quality_retention_regressions"
            ),
        },
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }
    report["review_hash"] = _sha256_json(
        {
            "selected_child_checkpoint": report["selected_child_checkpoint"],
            "prompt_retention": report["prompt_retention"],
            "learning_retention": report["learning_retention"],
            "sustained_decode": report["sustained_decode"],
            "review_gate": report["review_gate"],
        }
    )
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Language Quality Retention Review Bundle",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-replay-evidence", type=Path, required=True)
    parser.add_argument("--benchmark-suite-evidence", type=Path)
    parser.add_argument(
        "--projection-ablation-evidence",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--raw-sample-limit", type=int, default=12)
    args = parser.parse_args()

    build_language_quality_retention_review_bundle(
        output_path=args.output,
        quality_replay_evidence_path=args.quality_replay_evidence,
        benchmark_suite_evidence_path=args.benchmark_suite_evidence,
        projection_ablation_evidence_paths=tuple(args.projection_ablation_evidence),
        base_dir=args.base_dir,
        raw_sample_limit=args.raw_sample_limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
