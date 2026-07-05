from __future__ import annotations

import hashlib
import json

from marulho.evaluation.language_checkpoint_promotion_review import (
    SURFACE,
    build_language_checkpoint_promotion_review,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ready_inputs(tmp_path):
    original_parent = tmp_path / "original-parent.pt"
    evolved_child = tmp_path / "evolved-child.pt"
    selected_child = tmp_path / "selected-child.pt"
    original_parent.write_bytes(b"original-parent")
    evolved_child.write_bytes(b"evolved-child")
    selected_child.write_bytes(b"selected-child")
    original_hash = _sha256_bytes(b"original-parent")
    evolved_hash = _sha256_bytes(b"evolved-child")
    selected_hash = _sha256_bytes(b"selected-child")

    quality_path = tmp_path / "quality.json"
    evolution_path = tmp_path / "evolution.json"
    suite_path = tmp_path / "suite.json"
    _write_json(
        quality_path,
        {
            "surface": "marulho_language_quality_replay_experiment.v1",
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "active_language_path": "marulho_lm_head",
            "parent_checkpoint_path": str(evolved_child.name),
            "parent_checkpoint_sha256": evolved_hash,
            "child_checkpoint_path": str(selected_child.name),
            "child_checkpoint_sha256": selected_hash,
            "checkpoint_lineage": {
                "mutates_parent_checkpoint": False,
            },
            "candidate_selection": {
                "selected_candidate_id": "candidate-03",
                "selected_child_checkpoint_path": str(selected_child.name),
                "selected_child_checkpoint_sha256": selected_hash,
                "mutates_parent_checkpoint": False,
                "heldout_cases_used_for_replay_training": False,
                "candidates": [
                    {
                        "candidate_id": "candidate-03",
                        "selected": True,
                        "child_checkpoint_path": str(selected_child.name),
                        "child_checkpoint_sha256": selected_hash,
                        "learning_config": {
                            "learning_rate": 0.0005,
                            "replay_loss_weight": 1.5,
                            "max_steps": 6,
                        },
                        "update_tokens_per_second": 3042.957,
                        "total_window_tokens_per_second": 2735.45,
                    }
                ],
            },
            "experiment_review": {
                "same_child_generation_coherence_available": True,
                "heldout_generation_coherence_available": True,
                "heldout_generation_regressed_prompt_count": 0,
                "same_child_sustained_runtime_success": True,
                "same_child_controlled_decode_sustained_runtime_success": True,
                "records_controlled_decode_house_scale_sustained_runtime": True,
            },
            "heldout_prompt_suite": {
                "not_used_for_replay_training": True,
            },
            "generation_coherence_after": {
                "summary": {
                    "passed_case_count": 4,
                    "case_count": 4,
                    "mean_prefix_match_chars": 35.75,
                }
            },
            "heldout_generation_coherence_after": {
                "summary": {
                    "passed_case_count": 4,
                    "case_count": 4,
                    "mean_prefix_match_chars": 27.5,
                }
            },
            "sustained_runtime_evidence_summary": {
                "target_tokens": [8192, 131072, 524288],
                "min_tokens_per_second": 5343.657,
                "max_tokens_per_second": 8136.788,
            },
        },
    )
    _write_json(
        evolution_path,
        {
            "surface": "marulho_language_checkpoint_evolution_experiment.v1",
            "parent_checkpoint_path": str(original_parent.name),
            "child_final_checkpoint_path": str(evolved_child.name),
            "child_final_checkpoint_sha256": evolved_hash,
            "checkpoint_lineage": {
                "parent_checkpoint_sha256": original_hash,
            },
            "promotion_gate": {
                "checkpoint_lineage_complete": True,
                "rollback_to_parent_verified": True,
                "parent_runtime_unchanged": True,
                "child_checkpoint_available": True,
                "promotes_parent_promotion": False,
            },
        },
    )
    _write_json(
        suite_path,
        {
            "surface": "marulho_language_runtime_benchmark_suite.v1",
            "failed_category_count": 0,
            "missing_category_count": 0,
            "categories": [
                {"name": "gpu_kernel_correctness", "status": "pass"},
            ],
            "promotion_gate": {
                "status": "ready_for_review",
                "failed_category_names": [],
                "missing_required_category_names": [],
                "checkpoint_evolution_evidence_available": True,
                "quality_replay_evidence_available": True,
                "generation_coherence_available": True,
                "long_run_evidence_available": True,
                "controlled_decode_house_scale_evidence_available": True,
                "promotes_runtime_claim": False,
            },
        },
    )
    return quality_path, evolution_path, suite_path


def test_language_checkpoint_promotion_review_ready_for_operator_review(tmp_path):
    quality_path, evolution_path, suite_path = _write_ready_inputs(tmp_path)
    output = tmp_path / "promotion-review.json"

    report = build_language_checkpoint_promotion_review(
        output_path=output,
        quality_replay_evidence_path=quality_path,
        checkpoint_evolution_evidence_path=evolution_path,
        benchmark_suite_evidence_path=suite_path,
        base_dir=tmp_path,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["ready"] is True
    assert report["promotion_gate"]["eligible_for_operator_parent_promotion_review"] is True
    assert report["promotion_gate"]["eligible_for_live_parent_replacement"] is False
    assert report["promotion_gate"]["writes_live_checkpoint"] is False
    assert report["promotion_gate"]["mutates_runtime_state"] is False
    assert report["candidate_checkpoint"]["checkpoint_hash_verified"] is True
    assert report["lineage"]["quality_parent_matches_evolved_child_hash"] is True
    assert report["lineage"]["rollback_checkpoint_path"] == "evolved-child.pt"
    assert report["selected_child_evidence"]["update_tokens_per_second"] == 3042.957
    assert report["promotion_gate"]["missing_evidence"] == []
    assert (tmp_path / "README.md").exists()


def test_language_checkpoint_promotion_review_blocks_hash_mismatch(tmp_path):
    quality_path, evolution_path, suite_path = _write_ready_inputs(tmp_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["candidate_selection"]["selected_child_checkpoint_sha256"] = "wrong"
    quality["child_checkpoint_sha256"] = "wrong"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    report = build_language_checkpoint_promotion_review(
        output_path=tmp_path / "promotion-review.json",
        quality_replay_evidence_path=quality_path,
        checkpoint_evolution_evidence_path=evolution_path,
        benchmark_suite_evidence_path=suite_path,
        base_dir=tmp_path,
    )

    assert report["ready"] is False
    assert report["status"] == "blocked_missing_parent_promotion_review_evidence"
    assert report["promotion_gate"]["eligible_for_operator_parent_promotion_review"] is False
    assert "quality_selected_child_hash_matches_file" in report["promotion_gate"][
        "missing_evidence"
    ]
