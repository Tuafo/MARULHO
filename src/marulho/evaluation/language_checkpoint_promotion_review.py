"""Build a review packet for controlled MARULHO LM checkpoint promotion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_checkpoint_promotion_review.v1"
ARTIFACT_KIND = "marulho_language_checkpoint_promotion_review"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []


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


def _category_statuses(suite_report: Mapping[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for category in _list(suite_report.get("categories")):
        item = _mapping(category)
        name = str(item.get("name") or "")
        if name:
            statuses[name] = str(item.get("status") or "")
    return statuses


def _selected_candidate(
    quality_replay_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    selection = _mapping(quality_replay_report.get("candidate_selection"))
    selected_id = str(selection.get("selected_candidate_id") or "")
    for candidate in _list(selection.get("candidates")):
        item = _mapping(candidate)
        if bool(item.get("selected")) or str(item.get("candidate_id") or "") == selected_id:
            return item
    return {}


def _selected_checkpoint_path(quality_replay_report: Mapping[str, Any]) -> str:
    selection = _mapping(quality_replay_report.get("candidate_selection"))
    return str(
        selection.get("selected_child_checkpoint_path")
        or quality_replay_report.get("child_checkpoint_path")
        or ""
    )


def _selected_checkpoint_sha256(quality_replay_report: Mapping[str, Any]) -> str:
    selection = _mapping(quality_replay_report.get("candidate_selection"))
    return str(
        selection.get("selected_child_checkpoint_sha256")
        or quality_replay_report.get("child_checkpoint_sha256")
        or ""
    )


def build_language_checkpoint_promotion_review(
    *,
    output_path: str | Path,
    quality_replay_evidence_path: str | Path,
    checkpoint_evolution_evidence_path: str | Path,
    benchmark_suite_evidence_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify ready-for-review evidence for a child LM checkpoint.

    The report is an operator-review packet. It never installs the candidate,
    never writes a live checkpoint, and never promotes a runtime claim.
    """

    output = Path(output_path)
    quality_path = Path(quality_replay_evidence_path)
    evolution_path = Path(checkpoint_evolution_evidence_path)
    suite_path = Path(benchmark_suite_evidence_path)
    quality = _read_json(quality_path)
    evolution = _read_json(evolution_path)
    suite = _read_json(suite_path)

    suite_gate = _mapping(suite.get("promotion_gate"))
    quality_review = _mapping(quality.get("experiment_review"))
    quality_selection = _mapping(quality.get("candidate_selection"))
    quality_lineage = _mapping(quality.get("checkpoint_lineage"))
    evolution_gate = _mapping(evolution.get("promotion_gate"))
    evolution_lineage = _mapping(evolution.get("checkpoint_lineage"))
    categories = _category_statuses(suite)
    selected_candidate = _selected_candidate(quality)

    selected_path_text = _selected_checkpoint_path(quality)
    selected_hash = _selected_checkpoint_sha256(quality)
    selected_path = _resolve_artifact_path(selected_path_text, base_dir=base_dir)
    selected_file_exists = selected_path.is_file()
    selected_file_hash = _sha256_file(selected_path) if selected_file_exists else ""

    quality_parent_path = str(quality.get("parent_checkpoint_path") or "")
    quality_parent_hash = str(quality.get("parent_checkpoint_sha256") or "")
    evolution_child_path = str(evolution.get("child_final_checkpoint_path") or "")
    evolution_child_hash = str(evolution.get("child_final_checkpoint_sha256") or "")
    evolution_parent_path = str(evolution.get("parent_checkpoint_path") or "")
    evolution_parent_hash = str(
        evolution_lineage.get("parent_checkpoint_sha256")
        or evolution.get("parent_checkpoint_sha256")
        or ""
    )

    sustained_summary = _mapping(quality.get("sustained_runtime_evidence_summary"))
    selected_learning_config = _mapping(selected_candidate.get("learning_config"))
    selected_metrics = {
        "candidate_id": str(quality_selection.get("selected_candidate_id") or ""),
        "learning_rate": selected_learning_config.get("learning_rate"),
        "replay_loss_weight": selected_learning_config.get("replay_loss_weight"),
        "max_steps": selected_learning_config.get("max_steps"),
        "update_tokens_per_second": selected_candidate.get("update_tokens_per_second"),
        "total_window_tokens_per_second": selected_candidate.get(
            "total_window_tokens_per_second"
        ),
        "trained_prompt_passed_case_count": _mapping(
            _mapping(quality.get("generation_coherence_after")).get("summary")
        ).get("passed_case_count"),
        "trained_prompt_case_count": _mapping(
            _mapping(quality.get("generation_coherence_after")).get("summary")
        ).get("case_count"),
        "trained_mean_prefix_match_chars": _mapping(
            _mapping(quality.get("generation_coherence_after")).get("summary")
        ).get("mean_prefix_match_chars"),
        "heldout_prompt_passed_case_count": _mapping(
            _mapping(quality.get("heldout_generation_coherence_after")).get("summary")
        ).get("passed_case_count"),
        "heldout_prompt_case_count": _mapping(
            _mapping(quality.get("heldout_generation_coherence_after")).get("summary")
        ).get("case_count"),
        "heldout_mean_prefix_match_chars": _mapping(
            _mapping(quality.get("heldout_generation_coherence_after")).get("summary")
        ).get("mean_prefix_match_chars"),
        "sustained_targets": sustained_summary.get("target_tokens"),
        "min_sustained_tokens_per_second": sustained_summary.get(
            "min_tokens_per_second"
        ),
        "max_sustained_tokens_per_second": sustained_summary.get(
            "max_tokens_per_second"
        ),
    }

    required = {
        "quality_replay_surface_valid": quality.get("surface")
        == "marulho_language_quality_replay_experiment.v1",
        "checkpoint_evolution_surface_valid": evolution.get("surface")
        == "marulho_language_checkpoint_evolution_experiment.v1",
        "benchmark_suite_surface_valid": suite.get("surface")
        == "marulho_language_runtime_benchmark_suite.v1",
        "suite_ready_for_review": suite_gate.get("status") == "ready_for_review",
        "suite_failed_categories_absent": not bool(
            suite_gate.get("failed_category_names")
        )
        and int(suite.get("failed_category_count", 0) or 0) == 0,
        "suite_missing_required_categories_absent": not bool(
            suite_gate.get("missing_required_category_names")
        )
        and int(suite.get("missing_category_count", 0) or 0) == 0,
        "suite_checkpoint_evolution_available": bool(
            suite_gate.get("checkpoint_evolution_evidence_available")
        ),
        "suite_quality_replay_available": bool(
            suite_gate.get("quality_replay_evidence_available")
        ),
        "suite_generation_coherence_available": bool(
            suite_gate.get("generation_coherence_available")
        ),
        "suite_long_run_evidence_available": bool(
            suite_gate.get("long_run_evidence_available")
        ),
        "suite_controlled_house_scale_available": bool(
            suite_gate.get("controlled_decode_house_scale_evidence_available")
        ),
        "suite_gpu_kernel_correctness_passed": categories.get(
            "gpu_kernel_correctness"
        )
        == "pass",
        "suite_runtime_claim_not_promoted": suite_gate.get("promotes_runtime_claim")
        is False,
        "quality_selected_candidate_available": bool(selected_candidate),
        "quality_selected_child_checkpoint_available": selected_file_exists,
        "quality_selected_child_hash_matches_file": bool(selected_hash)
        and selected_file_hash == selected_hash,
        "quality_parent_not_mutated": _mapping(quality_lineage).get(
            "mutates_parent_checkpoint"
        )
        is False
        and quality_selection.get("mutates_parent_checkpoint") is False,
        "quality_same_child_generation_coherence": bool(
            quality_review.get("same_child_generation_coherence_available")
        ),
        "quality_heldout_generation_coherence": bool(
            quality_review.get("heldout_generation_coherence_available")
        ),
        "quality_heldout_regression_absent": int(
            quality_review.get("heldout_generation_regressed_prompt_count", 0) or 0
        )
        == 0,
        "quality_same_child_sustained_runtime_success": bool(
            quality_review.get("same_child_sustained_runtime_success")
        ),
        "quality_controlled_house_scale_sustained_success": bool(
            quality_review.get(
                "same_child_controlled_decode_sustained_runtime_success"
            )
        )
        and bool(quality_review.get("records_controlled_decode_house_scale_sustained_runtime")),
        "quality_uses_separate_heldout_prompts": _mapping(
            quality.get("heldout_prompt_suite")
        ).get("not_used_for_replay_training")
        is True
        and quality_selection.get("heldout_cases_used_for_replay_training") is False,
        "checkpoint_evolution_lineage_complete": bool(
            evolution_gate.get("checkpoint_lineage_complete")
        ),
        "checkpoint_evolution_rollback_verified": bool(
            evolution_gate.get("rollback_to_parent_verified")
        ),
        "checkpoint_evolution_parent_runtime_unchanged": bool(
            evolution_gate.get("parent_runtime_unchanged")
        ),
        "checkpoint_evolution_child_available": bool(
            evolution_gate.get("child_checkpoint_available")
        ),
        "checkpoint_evolution_does_not_promote_parent": evolution_gate.get(
            "promotes_parent_promotion"
        )
        is False,
        "quality_parent_matches_evolved_child_hash": bool(quality_parent_hash)
        and bool(evolution_child_hash)
        and quality_parent_hash == evolution_child_hash,
    }
    missing = [name for name, passed in required.items() if not bool(passed)]
    ready = not missing
    status = (
        "ready_for_operator_parent_promotion_review"
        if ready
        else "blocked_missing_parent_promotion_review_evidence"
    )

    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ready": ready,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": str(
            quality.get("active_language_path")
            or suite.get("active_language_path")
            or "marulho_lm_head"
        ),
        "output_path": str(output),
        "quality_replay_evidence_path": str(quality_path),
        "checkpoint_evolution_evidence_path": str(evolution_path),
        "benchmark_suite_evidence_path": str(suite_path),
        "candidate_checkpoint": {
            "surface": "marulho_language_checkpoint_promotion_candidate.v1",
            "selected_candidate_id": selected_metrics["candidate_id"],
            "checkpoint_path": selected_path_text,
            "checkpoint_sha256": selected_hash,
            "checkpoint_file_exists": selected_file_exists,
            "checkpoint_file_sha256": selected_file_hash,
            "checkpoint_hash_verified": bool(
                selected_hash and selected_file_hash == selected_hash
            ),
        },
        "lineage": {
            "surface": "marulho_language_checkpoint_promotion_lineage.v1",
            "original_parent_checkpoint_path": evolution_parent_path,
            "original_parent_checkpoint_sha256": evolution_parent_hash,
            "evolved_child_parent_checkpoint_path": quality_parent_path,
            "evolved_child_parent_checkpoint_sha256": quality_parent_hash,
            "evolution_child_checkpoint_path": evolution_child_path,
            "evolution_child_checkpoint_sha256": evolution_child_hash,
            "selected_quality_child_checkpoint_path": selected_path_text,
            "selected_quality_child_checkpoint_sha256": selected_hash,
            "quality_parent_matches_evolved_child_hash": required[
                "quality_parent_matches_evolved_child_hash"
            ],
            "parent_runtime_unchanged_during_evolution": required[
                "checkpoint_evolution_parent_runtime_unchanged"
            ],
            "rollback_to_evolution_parent_verified": required[
                "checkpoint_evolution_rollback_verified"
            ],
            "rollback_checkpoint_path": quality_parent_path,
            "rollback_checkpoint_sha256": quality_parent_hash,
        },
        "selected_child_evidence": selected_metrics,
        "benchmark_gate": {
            "surface": "marulho_language_checkpoint_promotion_benchmark_gate.v1",
            "suite_status": suite_gate.get("status"),
            "category_statuses": categories,
            "failed_category_names": list(suite_gate.get("failed_category_names") or []),
            "missing_required_category_names": list(
                suite_gate.get("missing_required_category_names") or []
            ),
            "promotes_runtime_claim": suite_gate.get("promotes_runtime_claim"),
        },
        "promotion_gate": {
            "surface": "marulho_language_checkpoint_promotion_gate.v1",
            "status": status,
            "required_evidence": required,
            "missing_evidence": missing,
            "eligible_for_operator_parent_promotion_review": ready,
            "eligible_for_live_parent_replacement": False,
            "operator_approval_recorded": False,
            "writes_live_checkpoint": False,
            "mutates_parent_checkpoint": False,
            "mutates_runtime_state": False,
            "runs_training": False,
            "runs_generation": False,
            "promotes_parent_promotion": False,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "next_gate": (
                "operator_review_parent_checkpoint_installation"
                if ready
                else "collect_missing_parent_promotion_review_evidence"
            ),
        },
    }
    report["review_hash"] = _sha256_json(
        {
            "candidate_checkpoint": report["candidate_checkpoint"],
            "lineage": report["lineage"],
            "promotion_gate": report["promotion_gate"],
        }
    )
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-replay-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-evolution-evidence", type=Path, required=True)
    parser.add_argument("--benchmark-suite-evidence", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    report = build_language_checkpoint_promotion_review(
        output_path=args.output,
        quality_replay_evidence_path=args.quality_replay_evidence,
        checkpoint_evolution_evidence_path=args.checkpoint_evolution_evidence,
        benchmark_suite_evidence_path=args.benchmark_suite_evidence,
        base_dir=args.base_dir,
    )
    return 0 if bool(report.get("ready")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
