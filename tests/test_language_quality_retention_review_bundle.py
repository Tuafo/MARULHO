from __future__ import annotations

import hashlib
import json

from marulho.evaluation.language_quality_retention_review_bundle import (
    SURFACE,
    build_language_quality_retention_review_bundle,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _coherence_report(checkpoint_path: str, prompt: str, *, passed: bool = True):
    continuation = " continued source text"
    return {
        "surface": "marulho_language_generation_coherence_report.v1",
        "checkpoint_path": checkpoint_path,
        "promotion_gate": {"generation_coherence_available": True},
        "summary": {
            "surface": "marulho_language_generation_coherence_summary.v1",
            "case_count": 1,
            "passed_case_count": 1 if passed else 0,
            "case_pass_rate": 1.0 if passed else 0.0,
            "mean_prefix_match_chars": 16.0 if passed else 0.0,
            "mean_prefix_match_fraction": 1.0 if passed else 0.0,
            "mean_source_continuation_loss": 0.5 if passed else 0.9,
            "mean_source_continuation_perplexity": 1.65 if passed else 2.45,
            "source_continuation_loss_available": True,
        },
        "cases": [
            {
                "prompt_text": prompt,
                "expected_source_continuation": continuation,
                "continuation_text": continuation,
                "passed": passed,
                "failure_reasons": [] if passed else ["source_prefix_match_below_threshold"],
                "prefix_match_chars": 16 if passed else 0,
                "prefix_match_fraction": 1.0 if passed else 0.0,
                "printable_fraction": 1.0,
                "distinct_bigram_fraction": 1.0,
                "source_continuation_loss": {
                    "loss": 0.5 if passed else 0.9,
                    "perplexity": 1.65 if passed else 2.45,
                },
            }
        ],
    }


def _delta(*, regressed: int = 0, loss_delta: float = -0.1):
    return {
        "surface": "marulho_language_quality_replay_coherence_delta.v1",
        "source_continuation_loss_available": True,
        "passed_case_count_delta": 0,
        "case_pass_rate_delta": 0.0,
        "mean_source_continuation_loss_delta": loss_delta,
        "mean_source_continuation_loss_regressed": loss_delta > 0.0,
        "mean_source_continuation_perplexity_delta": loss_delta,
        "mean_source_continuation_perplexity_regressed": loss_delta > 0.0,
        "prompt_pass_nonregressed": regressed == 0,
        "prompt_pass_nonregressed_but_loss_regressed": regressed == 0
        and loss_delta > 0.0,
        "repaired_prompt_count": 0,
        "repaired_prompts": [],
        "regressed_prompt_count": regressed,
        "regressed_prompts": ["prompt"] if regressed else [],
        "promotes_generation_quality_claim": False,
    }


def _ready_quality_report(tmp_path):
    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    child_hash = _sha256_bytes(b"child")
    parent_hash = _sha256_bytes(b"parent")
    child_name = child.name
    parent_name = parent.name
    return {
        "artifact_kind": "marulho_language_quality_replay_experiment",
        "surface": "marulho_language_quality_replay_experiment.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": "marulho_lm_head",
        "parent_checkpoint_path": parent_name,
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_path": child_name,
        "child_checkpoint_sha256": child_hash,
        "checkpoint_lineage": {
            "surface": "marulho_language_quality_replay_checkpoint_lineage.v1",
            "parent_checkpoint_path": parent_name,
            "child_checkpoint_path": child_name,
            "parent_checkpoint_sha256": parent_hash,
            "child_checkpoint_sha256": child_hash,
            "writes_child_checkpoint": True,
            "mutates_parent_checkpoint": False,
            "rollback_available": True,
        },
        "learning_evidence": {
            "surface": "marulho_language_continual_learning_window.v1",
            "status": "accepted_online_update",
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "active_language_path": "marulho_lm_head",
            "promotion_gate": {
                "eligible_for_online_learning_review": True,
                "rollback_available": True,
            },
            "rollback_evidence": {"restore_verified": True},
            "learning_evidence": {
                "update_token_count": 524288,
                "tokens_per_second": 5000.0,
                "total_window_tokens_per_second": 4500.0,
                "new_domain_loss_delta": -1.0,
                "old_domain_forgetting": 0.0,
                "general_replay_retention_delta": -0.5,
                "active_compute": {
                    "surface": "marulho_language_continual_active_compute.v1",
                    "active_columns_per_token": 4,
                    "total_columns": 16,
                    "runs_all_columns": False,
                },
            },
        },
        "generation_coherence_before": _coherence_report(parent_name, "train"),
        "generation_coherence_after": _coherence_report(child_name, "train"),
        "generation_coherence_delta": _delta(),
        "heldout_generation_coherence_before": _coherence_report(parent_name, "heldout"),
        "heldout_generation_coherence_after": _coherence_report(child_name, "heldout"),
        "heldout_generation_coherence_delta": _delta(),
        "fresh_heldout_generation_coherence_before": _coherence_report(
            parent_name,
            "fresh",
        ),
        "fresh_heldout_generation_coherence_after": _coherence_report(
            child_name,
            "fresh",
        ),
        "fresh_heldout_generation_coherence_delta": _delta(),
        "heldout_prompt_suite": {
            "not_used_for_replay_training": True,
            "training_prompt_overlap_count": 0,
        },
        "fresh_heldout_prompt_suite": {
            "enabled": True,
            "built_after_candidate_selection": True,
            "not_used_for_replay_training": True,
            "training_prompt_overlap_count": 0,
            "fixed_heldout_prompt_overlap_count": 0,
        },
        "candidate_selection": {
            "surface": "marulho_language_quality_replay_candidate_selection.v1",
            "selected_candidate_id": "candidate-00",
            "selected_child_checkpoint_path": child_name,
            "selected_child_checkpoint_sha256": child_hash,
            "candidate_count": 1,
            "selection_policy": "test_policy",
            "saves_child_checkpoint_per_candidate": True,
            "runs_sustained_runtime_only_for_selected_child": True,
            "mutates_parent_checkpoint": False,
            "heldout_cases_used_for_replay_training": False,
            "heldout_training_prompt_overlap_count": 0,
            "heldout_training_prompt_overlaps": [],
            "fresh_heldout_cases_used_for_replay_training": False,
            "fresh_heldout_training_prompt_overlap_count": 0,
            "fresh_heldout_training_prompt_overlaps": [],
            "fresh_heldout_fixed_prompt_overlap_count": 0,
            "fresh_heldout_fixed_prompt_overlaps": [],
            "selected_quality_retention_review": {
                "surface": "marulho_language_quality_replay_candidate_quality_retention.v1",
                "available": True,
                "suspicious": False,
                "suspicious_reasons": [],
                "trained_prompt_pass_nonregressed_but_loss_regressed": False,
                "heldout_prompt_pass_nonregressed_but_loss_regressed": False,
                "trained_source_continuation_loss_delta": -0.1,
                "heldout_source_continuation_loss_delta": -0.1,
                "promotes_generation_quality_claim": False,
            },
            "candidates": [
                {
                    "candidate_id": "candidate-00",
                    "selected": True,
                    "child_checkpoint_path": child_name,
                    "child_checkpoint_sha256": child_hash,
                    "learning_config": {
                        "learning_rate": 0.001,
                        "replay_loss_weight": 0.5,
                        "replay_gradient_projection_mode": "disabled",
                    },
                    "trained_generation_coherence_delta": _delta(),
                    "heldout_generation_coherence_delta": _delta(),
                    "quality_retention_review": {
                        "suspicious": False,
                    },
                    "update_tokens_per_second": 5000.0,
                    "total_window_tokens_per_second": 4500.0,
                }
            ],
        },
        "quality_generalization_review": {
            "heldout_prompt_coherence_available": True,
            "heldout_regressed_prompt_count": 0,
            "fresh_heldout_prompt_coherence_available": True,
            "fresh_heldout_regressed_prompt_count": 0,
            "same_child_sustained_runtime_success": True,
            "same_child_controlled_decode_sustained_runtime_success": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
        "sustained_runtime_evidence_summary": {
            "enabled": True,
            "all_success": True,
            "controlled_decode_all_success": True,
            "house_scale_524288_available": True,
            "controlled_decode_house_scale_524288_available": True,
            "target_tokens": [524288],
            "report_count": 1,
            "reports": [
                {
                    "checkpoint_path": child_name,
                    "target_tokens": 524288,
                    "token_delta": 524288,
                    "tokens_per_second": 8200.0,
                    "success": True,
                    "backend": "torch_cuda_graph_burst_decode_controls",
                    "decode_controls_requested": True,
                }
            ],
        },
        "sustained_runtime_evidence_reports": [
            {
                "checkpoint_path": child_name,
                "target_tokens": 524288,
                "token_delta": 524288,
                "tokens_per_second": 8200.0,
                "success": True,
                "backend": "torch_cuda_graph_burst_decode_controls",
                "decode_controls_requested": True,
            }
        ],
    }


def test_quality_retention_review_bundle_ready_for_review(tmp_path):
    quality_path = tmp_path / "quality.json"
    _write_json(quality_path, _ready_quality_report(tmp_path))

    report = build_language_quality_retention_review_bundle(
        output_path=tmp_path / "review.json",
        quality_replay_evidence_path=quality_path,
        base_dir=tmp_path,
    )

    assert report["surface"] == SURFACE
    assert report["status"] == "ready_for_review"
    assert report["ready_for_review"] is True
    assert report["selected_child_checkpoint"]["hash_verified"] is True
    assert report["review_gate"]["missing_evidence"] == []
    assert report["review_gate"]["failed_quality_checks"] == []
    assert report["learning_retention"]["active_compute_recorded"] is True
    assert report["sustained_decode"]["same_child_house_scale_524288_report_count"] == 1
    assert report["prompt_retention"]["fresh_post_selection_heldout_prompt_bank"][
        "same_child_checkpoint"
    ] is True
    assert report["prompt_retention"]["trained_prompt_bank"][
        "raw_continuation_review"
    ]["sample_count"] == 2
    assert report["promotes_generation_quality_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_quality_retention_review_bundle_blocks_missing_fresh_heldout(tmp_path):
    quality = _ready_quality_report(tmp_path)
    quality["fresh_heldout_prompt_suite"]["enabled"] = False
    quality.pop("fresh_heldout_generation_coherence_after")
    quality.pop("fresh_heldout_generation_coherence_delta")
    quality_path = tmp_path / "quality-missing-fresh.json"
    _write_json(quality_path, quality)

    report = build_language_quality_retention_review_bundle(
        output_path=tmp_path / "review.json",
        quality_replay_evidence_path=quality_path,
        base_dir=tmp_path,
    )

    assert report["status"] == "blocked_missing_required_evidence"
    assert "fresh_heldout_prompt_after_available" in report["review_gate"][
        "missing_evidence"
    ]
    assert "fresh_heldout_built_after_candidate_selection" in report["review_gate"][
        "missing_evidence"
    ]


def test_quality_retention_review_bundle_blocks_suspicious_projection_ablation(tmp_path):
    quality = _ready_quality_report(tmp_path)
    selection = quality["candidate_selection"]
    disabled = selection["candidates"][0]
    dense = {
        **disabled,
        "candidate_id": "candidate-01",
        "selected": True,
        "learning_config": {
            **disabled["learning_config"],
            "replay_gradient_projection_mode": "dense_core",
        },
        "trained_generation_coherence_delta": _delta(loss_delta=0.2),
        "heldout_generation_coherence_delta": _delta(loss_delta=0.3),
        "quality_retention_review": {"suspicious": True},
        "update_tokens_per_second": 3200.0,
        "total_window_tokens_per_second": 3000.0,
    }
    disabled["selected"] = False
    disabled["learning_config"]["replay_gradient_projection_mode"] = "disabled"
    disabled["total_window_tokens_per_second"] = 4500.0
    selection["candidates"] = [disabled, dense]
    selection["selected_candidate_id"] = "candidate-01"
    selection["selected_quality_retention_review"] = {
        "surface": "marulho_language_quality_replay_candidate_quality_retention.v1",
        "suspicious": True,
        "suspicious_reasons": [
            "trained_prompt_pass_nonregressed_but_loss_regressed",
            "heldout_prompt_pass_nonregressed_but_loss_regressed",
        ],
        "trained_prompt_pass_nonregressed_but_loss_regressed": True,
        "heldout_prompt_pass_nonregressed_but_loss_regressed": True,
        "trained_source_continuation_loss_delta": 0.2,
        "heldout_source_continuation_loss_delta": 0.3,
        "promotes_generation_quality_claim": False,
    }
    quality["generation_coherence_delta"] = _delta(loss_delta=0.2)
    quality["heldout_generation_coherence_delta"] = _delta(loss_delta=0.3)
    quality["fresh_heldout_generation_coherence_delta"] = _delta(regressed=1)
    quality_path = tmp_path / "quality-projection.json"
    _write_json(quality_path, quality)

    report = build_language_quality_retention_review_bundle(
        output_path=tmp_path / "review.json",
        quality_replay_evidence_path=quality_path,
        projection_ablation_evidence_paths=(quality_path,),
        base_dir=tmp_path,
    )

    assert report["status"] == "blocked_quality_retention"
    assert "selected_candidate_not_suspicious" in report["review_gate"][
        "failed_quality_checks"
    ]
    assert "fresh_heldout_prompt_regression_absent" in report["review_gate"][
        "failed_quality_checks"
    ]
    projection = report["projection_ablation_summary"]
    assert projection["projection_promoted"] is False
    assert "dense_core_projection_total_window_slower" in projection[
        "projection_rejection_reasons"
    ]
    assert "selected_projection_candidate_suspicious" in projection[
        "projection_rejection_reasons"
    ]
