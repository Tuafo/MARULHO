from __future__ import annotations

import json
import os

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
    learning_dir = reports / "language_brain_continual_learning"
    structural_dir = reports / "language_brain_structural_plasticity"
    training_dir = reports / "language_training_experiments"
    repair_dir = reports / "language_brain_generation_repair"
    suite_dir.mkdir(parents=True)
    learning_dir.mkdir(parents=True)
    structural_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
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
                    {
                        "name": "continual_learning",
                        "passed": True,
                        "evidence": {
                            "brain_installed_continual_learning_evidence": {
                                "best_report": {
                                    "active_language_path": "marulho_lm_head",
                                    "runtime_owner": "MarulhoBrain",
                                    "training_surface": (
                                        "marulho_language_continual_learning_window.v1"
                                    ),
                                    "brain_surface": (
                                        "marulho_brain_language_learning_window.v1"
                                    ),
                                    "device": "cuda:0",
                                    "learning_status": "accepted_online_update",
                                    "update_token_count": 524288,
                                    "house_scale_update_tokens_reached": True,
                                    "tokens_per_second": 3079.87,
                                    "total_window_tokens_per_second": 2810.81,
                                    "new_domain_loss_delta": 4.71,
                                    "old_domain_forgetting": -4.02,
                                    "general_replay_retention_delta": -4.0,
                                    "final_parameter_delta_l2": 26.66,
                                    "learned_brain_checkpoint_path": (
                                        "reports/language_brain_continual_learning/"
                                        "learned-brain.pt"
                                    ),
                                    "learned_brain_checkpoint_restore_verified": True,
                                    "memory_slots_enabled": True,
                                    "memory_slot_bounded_path": True,
                                    "memory_slot_candidate_slots_scored": 4194304,
                                    "memory_slot_runs_all_slots": False,
                                    "memory_slot_retrieval_backend": (
                                        "torch_autograd_bounded_memory_slots"
                                    ),
                                    "post_learning_sustained_enabled": True,
                                    "post_learning_sustained_success": True,
                                    "post_learning_sustained_token_delta": 524288,
                                    "post_learning_sustained_tokens_per_second": 8132.27,
                                    "post_learning_sustained_backend": (
                                        "torch_cuda_graph_burst_decode_controls"
                                    ),
                                    "promotes_runtime_claim": False,
                                }
                            }
                        },
                    },
                    {
                        "name": "forgetting",
                        "passed": True,
                        "evidence": {
                            "brain_installed_forgetting_measured": True,
                            "brain_installed_old_domain_forgetting": -4.02,
                            "old_domain_forgetting_within_tolerance": True,
                        },
                    },
                    {
                        "name": "replay_recovery",
                        "passed": True,
                        "evidence": {
                            "brain_installed_replay_retention_measured": True,
                            "brain_installed_general_replay_retention_delta": -4.0,
                            "general_replay_retention_within_tolerance": True,
                            "brain_installed_memory_slot_candidate_slots_scored": 4194304,
                            "brain_installed_memory_slot_runs_all_slots": False,
                        },
                    },
                    {
                        "name": "active_compute",
                        "passed": True,
                        "evidence": {
                            "active_expert_count_per_token": 1,
                            "active_parameters_per_token_estimate": 10280,
                            "active_parameter_fraction_estimate": 0.697,
                            "total_parameters": 14744,
                        },
                    },
                    {
                        "name": "growth_prune_safety",
                        "passed": True,
                        "evidence": {
                            "brain_installed_structural_plasticity_evidence": {
                                "best_report": {
                                    "active_language_path": "marulho_lm_head",
                                    "runtime_owner": "MarulhoBrain",
                                    "brain_surface": (
                                        "marulho_brain_language_structural_transaction.v1"
                                    ),
                                    "training_surface": (
                                        "marulho_language_structural_plasticity_transaction.v1"
                                    ),
                                    "trace_event": "language_structure",
                                    "transaction_status": "applied_structural_mutation",
                                    "proposal_kind": "route_bank_expansion",
                                    "applied": True,
                                    "checkpoint_restore_verified": True,
                                    "rollback_verified": True,
                                    "heldout_non_regression": True,
                                    "source_expert_count": 18,
                                    "target_expert_count": 18,
                                    "source_memory_slot_count": 1024,
                                    "target_memory_slot_count": 1024,
                                    "memory_slot_count_delta": 0,
                                    "source_route_candidate_count": 8,
                                    "target_route_candidate_count": 12,
                                    "route_bank_candidate_count_delta": 4,
                                    "pre_structure_checkpoint_path": (
                                        "reports/language_brain_structural_plasticity/"
                                        "pre-structure-brain.pt"
                                    ),
                                    "pre_structure_checkpoint_restore_verified": True,
                                    "post_structure_checkpoint_path": (
                                        "reports/language_brain_structural_plasticity/"
                                        "post-structure-brain.pt"
                                    ),
                                    "post_structure_checkpoint_restore_verified": True,
                                    "post_structure_sustained_enabled": True,
                                    "post_structure_sustained_success": True,
                                    "post_structure_sustained_token_delta": 524288,
                                    "post_structure_sustained_tokens_per_second": 8060.86,
                                    "post_structure_sustained_backend": (
                                        "torch_cuda_graph_burst_decode_controls"
                                    ),
                                    "post_structure_sustained_triton_failure_count": 0,
                                    "status_read_mutation_absent": True,
                                    "promotes_runtime_claim": False,
                                }
                            },
                            "route_bank_proposal_mutates_runtime_state": False,
                            "route_bank_runs_all_columns": False,
                            "route_bank_source_candidate_count": 8,
                            "route_bank_target_candidate_count": 12,
                        },
                    },
                    {
                        "name": "checkpoint_restore",
                        "passed": True,
                        "evidence": {
                            "checkpoint_path": "reports/language_benchmark_suite/checkpoint.pt",
                            "brain_installed_pre_learning_checkpoint_path": (
                                "reports/language_brain_continual_learning/"
                                "pre-learning-brain.pt"
                            ),
                            "brain_installed_learned_checkpoint_path": (
                                "reports/language_brain_continual_learning/"
                                "learned-brain.pt"
                            ),
                            "brain_installed_learned_checkpoint_restore_verified": True,
                        },
                    },
                    {
                        "name": "rollback",
                        "passed": True,
                        "evidence": {
                            "checkpoint_lineage_complete": True,
                            "lineage_id": "lineage-1",
                            "parent_kept_installed": True,
                            "parent_runtime_unchanged": True,
                            "rollback_to_parent_verified": True,
                            "operator_review_required": True,
                            "long_run_evidence_required_for_promotion": True,
                            "saved_checkpoint_evolution_evidence": {
                                "checkpoint_evolution_evidence_available": False,
                                "promotes_runtime_claim": False,
                            },
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
    (learning_dir / "continual-learning.json").write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_brain_installed_continual_learning_evidence",
                "surface": "marulho_language_brain_installed_continual_learning_evidence.v1",
                "active_language_path": "marulho_lm_head",
                "runtime_owner": "MarulhoBrain",
                "external_llm_used": False,
                "service_owned_cognition": False,
                "status_read_mutation": False,
                "promotes_runtime_claim": False,
                "update_token_count": 524288,
                "tokens_per_second": 3079.87,
                "total_window_tokens_per_second": 2810.81,
                "learning_summary": {
                    "brain_surface": "marulho_brain_language_learning_window.v1",
                    "training_surface": "marulho_language_continual_learning_window.v1",
                    "device": "cuda:0",
                    "status": "accepted_online_update",
                    "mutates_language_model_weights": True,
                    "trace_event": "language_learn",
                    "update_token_count": 524288,
                    "tokens_per_second": 3079.87,
                    "total_window_tokens_per_second": 2810.81,
                    "new_domain_loss_delta": 4.71,
                    "old_domain_forgetting": -4.02,
                    "general_replay_retention_delta": -4.0,
                    "final_parameter_delta_l2": 26.66,
                    "memory_slots": {
                        "enabled": True,
                        "bounded_memory_slot_path": True,
                        "candidate_slots_scored": 4194304,
                        "runs_all_slots": False,
                        "memory_slot_retrieval_backend": (
                            "torch_autograd_bounded_memory_slots"
                        ),
                    },
                    "batch_device_staging": {
                        "surface": (
                            "marulho_language_continual_batch_device_staging.v1"
                        ),
                        "staged_before_measured_update_window": True,
                        "all_update_batches_on_device_before_timing": True,
                        "measured_update_loop_caller_device_transfer_calls": 0,
                    },
                    "measured_update_loop_caller_device_transfer_calls": 0,
                    "training_window_triton_accounting": {
                        "tracked_triton_failure_count": 0,
                        "tracked_torch_fallback_call_count": 0,
                    },
                },
                "learning_evidence": {
                    "training_window_triton_accounting": {
                        "surface": (
                            "marulho_language_continual_training_window_"
                            "triton_accounting.v1"
                        ),
                        "scope": "measured_update_window_only",
                        "tracked_triton_kernel_used_names": [
                            "language_rmsnorm_triton",
                            "language_plif_triton",
                        ],
                        "tracked_torch_fallback_calls": 512,
                        "tracked_triton_failure_count": 0,
                        "language_sampled_vocab_ce_triton": {
                            "torch_fallback_calls": 512,
                            "triton_failure_count": 0,
                        },
                        "language_memory_slots_triton": {
                            "torch_fallback_calls": 0,
                            "triton_failure_count": 0,
                        },
                    },
                },
                "learned_brain_checkpoint_path": (
                    "reports/language_brain_continual_learning/learned-brain.pt"
                ),
                "learned_brain_checkpoint": {"restore_verified": True},
                "post_learning_sustained_window": {
                    "enabled": True,
                    "success": True,
                    "target_tokens": 524288,
                    "token_delta": 524288,
                    "tokens_per_second": 8132.27,
                    "backend": "torch_cuda_graph_burst_decode_controls",
                    "tracked_triton_kernel_failure_count": 0,
                    "tracked_triton_kernel_used_names": [
                        "language_rmsnorm_triton",
                        "language_plif_triton",
                    ],
                },
                "promotion_gate": {
                    "house_scale_524288_update_tokens_reached": True,
                    "learned_brain_checkpoint_restore_verified": True,
                    "status_read_mutation_absent": True,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (structural_dir / "structural-plasticity.json").write_text(
        json.dumps(
            {
                "artifact_kind": (
                    "marulho_language_brain_installed_structural_plasticity_evidence"
                ),
                "surface": (
                    "marulho_language_brain_installed_structural_plasticity_evidence.v1"
                ),
                "active_language_path": "marulho_lm_head",
                "runtime_owner": "MarulhoBrain",
                "external_llm_used": False,
                "service_owned_cognition": False,
                "status_read_mutation": False,
                "promotes_runtime_claim": False,
                "promotes_generation_quality_claim": False,
                "structural_transaction_summary": {
                    "brain_surface": "marulho_brain_language_structural_transaction.v1",
                    "training_surface": (
                        "marulho_language_structural_plasticity_transaction.v1"
                    ),
                    "trace_event": "language_structure",
                    "status": "applied_structural_mutation",
                    "proposal_kind": "route_bank_expansion",
                    "applied": True,
                    "operator_approved": True,
                    "checkpoint_restore_verified": True,
                    "rollback_verified": True,
                    "heldout_non_regression": True,
                    "source_expert_count": 18,
                    "target_expert_count": 18,
                    "source_memory_slot_count": 1024,
                    "target_memory_slot_count": 1024,
                    "memory_slot_count_delta": 0,
                    "source_route_candidate_count": 8,
                    "target_route_candidate_count": 12,
                    "route_bank_candidate_count_delta": 4,
                },
                "pre_structural_brain_checkpoint": {
                    "path": (
                        "reports/language_brain_structural_plasticity/"
                        "pre-structure-brain.pt"
                    ),
                    "sha256": "pre-sha",
                    "restore_verified": True,
                },
                "post_structural_brain_checkpoint": {
                    "path": (
                        "reports/language_brain_structural_plasticity/"
                        "post-structure-brain.pt"
                    ),
                    "sha256": "post-sha",
                    "restore_verified": True,
                },
                "post_structure_sustained_window": {
                    "enabled": True,
                    "success": True,
                    "target_tokens": 524288,
                    "token_delta": 524288,
                    "tokens_per_second": 8060.86,
                    "backend": "torch_cuda_graph_burst_decode_controls",
                    "device": "cuda:0",
                    "tracked_triton_kernel_failure_count": 0,
                    "tracked_triton_kernel_used_names": [
                        "language_rmsnorm_triton",
                        "language_plif_triton",
                    ],
                },
                "promotion_gate": {
                    "proposal_non_mutating": True,
                    "proposal_runs_through_marulho_brain": True,
                    "structural_apply_runs_through_marulho_brain": True,
                    "records_checkpoint_backed_transaction": True,
                    "records_rollback_evidence": True,
                    "pre_structure_brain_checkpoint_restore_verified": True,
                    "post_structure_brain_checkpoint_restore_verified": True,
                    "status_read_mutation_absent": True,
                    "promotes_runtime_claim": False,
                    "promotes_generation_quality_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (training_dir / "state-block-impact.json").write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_state_block_runtime_impact",
                "surface": "marulho_language_state_block_runtime_impact.v1",
                "comparison": {
                    "baseline_tokens_per_second": 12321.43,
                    "preallocated_tokens_per_second": 12068.36,
                    "preallocated_vs_baseline_tokens_per_second_ratio": 0.979,
                    "parity_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (training_dir / "eligibility-impact.json").write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_eligibility_trace_runtime_impact",
                "surface": "marulho_language_eligibility_trace_runtime_impact.v1",
                "comparison": {
                    "baseline_tokens_per_second": 12760.57,
                    "deferred_tokens_per_second": 12148.41,
                    "deferred_vs_baseline_tokens_per_second_ratio": 0.952,
                    "parity_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    memory_slot_training_report = training_dir / "memory-slot-training-impact.json"
    memory_slot_training_report.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_memory_slot_training_impact",
                "surface": "marulho_language_memory_slot_training_impact.v1",
                "report_status": "final",
                "review": {
                    "hot_update_evidence_mode": "post_window_telemetry_probe",
                    "per_step_evidence_dict_build": False,
                    "per_step_memory_slot_stats_delta": False,
                },
                "comparison": {
                    "control_tokens_per_second": 3171.11,
                    "bounded_tokens_per_second": 3076.58,
                    "triton_training_tokens_per_second": 3110.44,
                    "triton_training_vs_bounded_tokens_per_second_ratio": 1.011,
                    "triton_training_vs_control_tokens_per_second_ratio": 0.981,
                    "bounded_avoids_all_slot_scan": True,
                },
            }
        ),
        encoding="utf-8",
    )
    partial_memory_slot_training_report = (
        training_dir / "memory-slot-training-impact-partial.json"
    )
    partial_memory_slot_training_report.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_memory_slot_training_impact",
                "surface": "marulho_language_memory_slot_training_impact.v1",
                "report_status": "partial",
                "partial_reason": "completed_control_arm",
                "completed_arm_names": ["memory_slots_disabled_control"],
                "missing_arm_names": [
                    "bounded_torch_memory_slots",
                    "triton_forward_torch_backward_memory_slots",
                ],
            }
        ),
        encoding="utf-8",
    )
    newer_mtime = memory_slot_training_report.stat().st_mtime + 10.0
    os.utime(partial_memory_slot_training_report, (newer_mtime, newer_mtime))

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
    assert projection["training_throughput_evidence"]["update_token_count"] == 524288
    assert projection["training_throughput_evidence"]["tokens_per_second"] == 3079.87
    assert projection["training_throughput_evidence"]["device"] == "cuda:0"
    assert projection["training_throughput_evidence"]["memory_slots"]["runs_all_slots"] is False
    assert projection["training_throughput_evidence"]["post_learning_sustained"][
        "tokens_per_second"
    ] == 8132.27
    assert projection["training_throughput_evidence"][
        "measured_update_loop_caller_device_transfer_calls"
    ] == 0
    assert projection["training_throughput_evidence"]["batch_device_staging"][
        "all_update_batches_on_device_before_timing"
    ] is True
    training_accounting = projection["training_throughput_evidence"][
        "training_window_triton_accounting"
    ]
    assert training_accounting["tracked_torch_fallback_calls"] == 512
    assert training_accounting["language_sampled_vocab_ce_triton"][
        "torch_fallback_calls"
    ] == 512
    assert projection["forgetting_replay_evidence"]["forgetting_measured"] is True
    assert projection["forgetting_replay_evidence"]["replay_retention_measured"] is True
    assert projection["active_compute_evidence"]["active_parameters_per_token_estimate"] == 10280
    assert projection["structural_plasticity_evidence"]["proposal_kind"] == (
        "route_bank_expansion"
    )
    assert projection["structural_plasticity_evidence"]["mutation"][
        "source_route_candidate_count"
    ] == 8
    assert projection["structural_plasticity_evidence"]["mutation"][
        "target_route_candidate_count"
    ] == 12
    assert projection["structural_plasticity_evidence"]["rollback_verified"] is True
    assert projection["structural_plasticity_evidence"]["post_structure_checkpoint"][
        "delete_protected_by_current_evidence"
    ] is True
    assert projection["structural_plasticity_evidence"]["post_structure_sustained"][
        "tokens_per_second"
    ] == 8060.86
    assert projection["checkpoint_lineage_evidence"][
        "structural_post_checkpoint_restore_verified"
    ] is True
    assert projection["checkpoint_lineage_evidence"]["checkpoint_evolution"][
        "checkpoint_lineage_complete"
    ] is True
    assert projection["house_scale_throughput_evidence"]["target_tokens"] == 524288
    assert projection["house_scale_throughput_evidence"]["house_scale_gate_reached"] is True
    assert projection["house_scale_throughput_evidence"]["tokens_per_second"] == 8123.13
    assert projection["gpu_kernel_evidence"]["generation_tracked_failure_count"] == 0
    decisions = {
        item["name"]: item for item in projection["backend_bottleneck_evidence"]["decisions"]
    }
    assert decisions["state_block_preallocation"]["status"] == "rejected_as_default"
    assert decisions["deferred_eligibility_trace_scan"][
        "candidate_vs_baseline_ratio"
    ] == 0.952
    assert decisions["memory_slot_triton_training_autograd"][
        "accepted_current_backend"
    ] == "torch_autograd_bounded_memory_slots"
    assert decisions["memory_slot_triton_training_autograd"]["report_status"] == "final"
    assert decisions["memory_slot_triton_training_autograd"][
        "hot_update_evidence_mode"
    ] == "post_window_telemetry_probe"
    assert decisions["memory_slot_triton_training_autograd"][
        "per_step_evidence_dict_build"
    ] is False
    assert decisions["memory_slot_triton_training_autograd"][
        "per_step_memory_slot_stats_delta"
    ] is False
    training_backend = projection["backend_bottleneck_evidence"][
        "current_training_window_backend_evidence"
    ]
    assert training_backend["tracked_torch_fallback_calls"] == 512
    assert training_backend["tracked_torch_fallback_kernel_names"] == [
        "language_sampled_vocab_ce_triton"
    ]
    assert training_backend["all_update_batches_on_device_before_timing"] is True
    assert training_backend[
        "measured_update_loop_caller_device_transfer_calls"
    ] == 0
    assert training_backend["gpu_training_hot_path_status"] == (
        "torch_fallbacks_present"
    )
    memory_slot_selection = projection["backend_bottleneck_evidence"][
        "memory_slot_training_report_selection"
    ]
    assert memory_slot_selection["latest_report_status"] == "partial"
    assert memory_slot_selection["latest_report_complete"] is False
    assert memory_slot_selection["latest_completed_arm_names"] == [
        "memory_slots_disabled_control"
    ]
    assert memory_slot_selection["latest_missing_arm_names"] == [
        "bounded_torch_memory_slots",
        "triton_forward_torch_backward_memory_slots",
    ]
    assert memory_slot_selection["backend_decision_report_status"] == "final"
    assert memory_slot_selection["backend_decision_uses_latest_report"] is False
    assert memory_slot_selection[
        "partial_reports_do_not_replace_complete_backend_decision"
    ] is True
    assert projection["current_checkpoint"]["path"] == checkpoint_path
    assert projection["current_checkpoint"]["delete_protected_by_current_evidence"] is True
