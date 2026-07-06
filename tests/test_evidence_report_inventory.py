from __future__ import annotations

import json

from marulho.reporting.evidence_inventory import (
    CURRENT_LANGUAGE_EVIDENCE_SURFACE,
    SURFACE,
    build_current_language_evidence_projection,
    build_evidence_report_inventory,
)


def test_evidence_report_inventory_summarizes_saved_reports_without_promotion(
    tmp_path,
) -> None:
    reports = tmp_path / "reports"
    suite_dir = reports / "language_benchmark_suite"
    suite_dir.mkdir(parents=True)
    report_path = suite_dir / "language-suite.json"
    report_path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_runtime_benchmark_suite",
                "surface": "marulho_language_runtime_benchmark_suite.v1",
                "success": True,
                "external_llm_used": False,
                "promotion_gate": {
                    "status": "blocked_missing_required_evidence",
                    "promotes_runtime_claim": False,
                    "missing_required_category_names": [
                        "grounding_support",
                        "gpu_kernel_correctness",
                    ],
                    "failed_category_names": [],
                },
            }
        ),
        encoding="utf-8",
    )
    invalid_path = reports / "invalid.json"
    invalid_path.write_text("{not json", encoding="utf-8")

    inventory = build_evidence_report_inventory(reports, limit=10)
    records = {item["relative_path"]: item for item in inventory["reports"]}

    assert inventory["surface"] == SURFACE
    assert inventory["reports_not_run_by_service"] is True
    assert inventory["mutates_runtime_state"] is False
    assert inventory["report_count"] == 2
    assert records["language_benchmark_suite/language-suite.json"]["readable"] is True
    assert records["language_benchmark_suite/language-suite.json"]["artifact_kind"] == (
        "marulho_language_runtime_benchmark_suite"
    )
    assert records["language_benchmark_suite/language-suite.json"]["promotion_status"] == (
        "blocked_missing_required_evidence"
    )
    assert records["language_benchmark_suite/language-suite.json"]["promotes_runtime_claim"] is False
    assert records["language_benchmark_suite/language-suite.json"][
        "missing_required_category_names"
    ] == ["grounding_support", "gpu_kernel_correctness"]
    assert records["invalid.json"]["readable"] is False
    assert "JSONDecodeError" in records["invalid.json"]["parse_error"]


def test_current_language_evidence_projection_tracks_selected_repair_without_running(
    tmp_path,
) -> None:
    reports = tmp_path / "reports"
    suite_dir = reports / "language_benchmark_suite"
    repair_dir = reports / "language_brain_generation_repair"
    suite_dir.mkdir(parents=True)
    repair_dir.mkdir(parents=True)
    checkpoint_path = (
        "reports/language_brain_generation_repair/"
        "selected-candidate-02-repaired-brain.pt"
    )
    suite_path = suite_dir / "language-suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_runtime_benchmark_suite",
                "surface": "marulho_language_runtime_benchmark_suite.v1",
                "external_llm_used": False,
                "promotion_gate": {
                    "status": "ready_for_review",
                    "promotes_runtime_claim": False,
                    "missing_required_category_names": [],
                    "failed_category_names": [],
                    "long_run_evidence_available": True,
                    "controlled_decode_house_scale_evidence_available": True,
                    "generation_controlled_decode_house_scale_aligned": True,
                    "brain_installed_generation_evidence_available": True,
                    "brain_installed_generation_repair_evidence_available": True,
                },
                "categories": [
                    {
                        "name": "generation_coherence",
                        "passed": True,
                        "evidence": {
                            "brain_installed_generation_evidence": {
                                "best_report": {
                                    "active_language_path": "marulho_lm_head",
                                    "runtime_owner": "MarulhoBrain",
                                    "brain_checkpoint_path": checkpoint_path,
                                    "brain_checkpoint_restore_verified": True,
                                    "case_count": 4,
                                    "passed_case_count": 4,
                                    "case_pass_rate": 1.0,
                                    "mean_prefix_match_chars": 34.0,
                                    "generation_runs_through_marulho_brain": True,
                                    "grounded_prompt_suite_available": True,
                                    "status_read_mutation_absent": True,
                                    "promotes_runtime_claim": False,
                                    "promotes_generation_quality_claim": False,
                                }
                            },
                            "brain_installed_generation_long_run_alignment": {
                                "same_checkpoint_controlled_decode_house_scale_available": True,
                                "matching_reports": [
                                    {
                                        "checkpoint_path": checkpoint_path,
                                        "runtime_owner": "MarulhoLanguageModel",
                                        "backend": "torch_cuda_graph_burst_decode_controls",
                                        "device": "cuda:0",
                                        "success": True,
                                        "report_status": "final",
                                        "target_tokens": 524288,
                                        "token_delta": 524288,
                                        "tokens_per_second": 8123.13,
                                        "triton_kernel_used": True,
                                        "promotes_runtime_claim": False,
                                    }
                                ],
                            },
                        },
                    },
                    {
                        "name": "gpu_kernel_correctness",
                        "passed": True,
                        "evidence": {
                            "covered_kernel_names": [
                                "language_rmsnorm_forward",
                                "language_plif_forward",
                            ]
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    repair_sweep_path = repair_dir / "repair-sweep.json"
    repair_sweep_path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_brain_installed_generation_repair_sweep",
                "surface": "marulho_language_brain_installed_generation_repair_sweep.v1",
                "active_language_path": "marulho_lm_head",
                "runtime_owner": "MarulhoBrain",
                "external_llm_used": False,
                "service_owned_cognition": False,
                "promotes_runtime_claim": False,
                "promotes_generation_quality_claim": False,
                "candidate_count": 3,
                "candidate_selection": {
                    "candidate_count": 3,
                    "selected_candidate_id": "candidate-02",
                    "selected_repaired_brain_checkpoint_path": checkpoint_path,
                    "selected_repaired_brain_checkpoint_sha256": "sha-02",
                    "runs_sustained_runtime_only_for_selected_child": True,
                },
                "selected_repair_evidence": {
                    "candidate_id": "candidate-02",
                    "repaired_brain_checkpoint_path": checkpoint_path,
                    "repaired_brain_checkpoint_sha256": "sha-02",
                    "repaired_brain_checkpoint_restore_verified": True,
                    "case_count": 4,
                    "pre_passed_case_count": 3,
                    "post_passed_case_count": 4,
                    "passed_case_count_delta": 1,
                    "mean_prefix_match_chars_delta": 10.5,
                    "regressed_prompt_count": 0,
                    "update_token_count": 491520,
                    "tokens_per_second": 2510.33,
                    "total_window_tokens_per_second": 2236.53,
                    "learning_config": {
                        "learning_rate": 0.0002,
                        "repair_pass_count": 3,
                    },
                },
                "post_repair_sustained_window": {
                    "runtime_owner": "MarulhoLanguageModel",
                    "active_language_path": "marulho_lm_head",
                    "checkpoint_path": checkpoint_path,
                    "backend": "torch_cuda_graph_burst_decode_controls",
                    "device": "cuda:0",
                    "success": True,
                    "report_status": "final",
                    "target_tokens": 524288,
                    "token_delta": 524288,
                    "tokens_per_second": 8123.13,
                    "triton_kernel_used": True,
                    "tracked_triton_kernel_failure_count": 0,
                    "tracked_triton_kernel_used_names": [
                        "language_rmsnorm_triton",
                        "language_plif_triton",
                    ],
                    "promotes_runtime_claim": False,
                },
                "promotion_gate": {
                    "selected_checkpoint_restore_verified": True,
                    "selected_post_repair_sustained_target_reached": True,
                    "promotes_runtime_claim": False,
                    "promotes_generation_quality_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )

    projection = build_current_language_evidence_projection(reports)

    assert projection["surface"] == CURRENT_LANGUAGE_EVIDENCE_SURFACE
    assert projection["reports_not_run_by_service"] is True
    assert projection["mutates_runtime_state"] is False
    assert projection["service_owned_cognition"] is False
    assert projection["runtime_review_gate"]["status"] == "ready_for_review"
    assert projection["runtime_review_gate"]["promotes_runtime_claim"] is False
    assert projection["generation_evidence"]["passed_case_count"] == 4
    assert projection["generation_evidence"]["generation_runs_through_marulho_brain"] is True
    assert projection["repair_evidence"]["selected_candidate_id"] == "candidate-02"
    assert projection["repair_evidence"]["regressed_prompt_count"] == 0
    assert projection["repair_evidence"]["update_token_count"] == 491520
    assert projection["house_scale_throughput_evidence"]["target_tokens"] == 524288
    assert projection["house_scale_throughput_evidence"]["house_scale_gate_reached"] is True
    assert projection["house_scale_throughput_evidence"]["tokens_per_second"] == 8123.13
    assert projection["gpu_kernel_evidence"]["generation_tracked_failure_count"] == 0
    assert projection["current_checkpoint"]["path"] == checkpoint_path
    assert projection["current_checkpoint"]["delete_protected_by_current_evidence"] is True
