from __future__ import annotations

import json

from marulho.evaluation.language_runtime_benchmark_suite import (
    BRAIN_INSTALLED_CONTINUAL_LEARNING_ARTIFACT_KIND,
    BRAIN_INSTALLED_CONTINUAL_LEARNING_SURFACE,
    ELIGIBILITY_TRACE_KERNEL_NAME,
    EXPERT_DISPATCH_KERNEL_NAME,
    CHECKPOINT_EVOLUTION_EXPERIMENT_ARTIFACT_KIND,
    CHECKPOINT_EVOLUTION_EXPERIMENT_SURFACE,
    GENERATION_COHERENCE_ARTIFACT_KIND,
    GENERATION_COHERENCE_SURFACE,
    KERNEL_ARTIFACT_KIND,
    KERNEL_SURFACE,
    MEMORY_SLOT_ARCHITECTURE_COST_SURFACE,
    MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
    MEMORY_SLOT_RUNTIME_IMPACT_ARTIFACT_KIND,
    MEMORY_SLOT_RUNTIME_IMPACT_SURFACE,
    PLIF_FORWARD_KERNEL_NAME,
    PLIF_SURROGATE_KERNEL_NAME,
    QUALITY_REPLAY_ARTIFACT_KIND,
    QUALITY_REPLAY_SURFACE,
    RMSNORM_KERNEL_NAME,
    ROUTE_TOPK_KERNEL_NAME,
    SAMPLED_VOCAB_CE_KERNEL_NAME,
    SELECTIVE_SCAN_KERNEL_NAME,
    STRUCTURAL_PLASTICITY_EXPERIMENT_ARTIFACT_KIND,
    STRUCTURAL_PLASTICITY_EXPERIMENT_SURFACE,
    STRUCTURAL_PLASTICITY_TRANSACTION_ARTIFACT_KIND,
    STRUCTURAL_PLASTICITY_TRANSACTION_SURFACE,
    SURFACE,
    SUSTAINED_ARTIFACT_KIND,
    SUSTAINED_SURFACE,
    run_language_runtime_benchmark_suite,
)


def _write_sustained_report(
    path,
    *,
    token_delta: int,
    controlled_decode: bool = False,
    checkpoint_path: str = "reports/language_training_experiments/checkpoint.pt",
) -> None:
    backend = (
        "torch_cuda_graph_burst_decode_controls"
        if controlled_decode
        else "torch_eager_cpu"
    )
    path.write_text(
        json.dumps(
            {
                "artifact_kind": SUSTAINED_ARTIFACT_KIND,
                "surface": SUSTAINED_SURFACE,
                "report_status": "final",
                "success": True,
                "target_tokens": token_delta,
                "token_delta": token_delta,
                "tokens_per_second": 1234.5,
                "checkpoint_path": checkpoint_path,
                "runtime_owner": "MarulhoLanguageModel",
                "active_language_path": "marulho_lm_head",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "device_backend": {
                    "device": "cuda:0" if controlled_decode else "cpu",
                    "backend": backend,
                    "cuda_graph_burst_used": controlled_decode,
                    "triton_kernel_used": False,
                    "promoted_hot_path": False,
                },
                "execution_evidence": {
                    "backend": backend,
                    "decode_controls_requested": controlled_decode,
                    "decode_controls_backend": "torch_device_tensor",
                    "decode_controls_cpu_token_copy": False,
                    "decode_controls_graph_compatible": controlled_decode,
                    "cuda_graph_decode_controls_used": controlled_decode,
                    "repetition_penalty": 1.15 if controlled_decode else 1.0,
                    "repetition_penalty_applied": controlled_decode,
                    "repetition_penalty_adjusted_token_count": (
                        4096 if controlled_decode else 0
                    ),
                    "no_repeat_ngram_size": 3 if controlled_decode else 0,
                    "no_repeat_ngram_applied": controlled_decode,
                    "no_repeat_ngram_banned_token_count": (
                        512 if controlled_decode else 0
                    ),
                    "decode_control_fallback_count": 0,
                },
                "generation_decode": {
                    "surface": "marulho_language_generation_decode_policy.v1",
                    "decode_strategy": "greedy_argmax",
                    "decode_controls_requested": controlled_decode,
                    "decode_controls_backend": "torch_device_tensor",
                    "decode_controls_cpu_token_copy": False,
                    "decode_controls_graph_compatible": controlled_decode,
                    "cuda_graph_decode_controls_used": controlled_decode,
                    "repetition_penalty": 1.15 if controlled_decode else 1.0,
                    "repetition_penalty_applied": controlled_decode,
                    "repetition_penalty_adjusted_token_count": (
                        4096 if controlled_decode else 0
                    ),
                    "no_repeat_ngram_size": 3 if controlled_decode else 0,
                    "no_repeat_ngram_applied": controlled_decode,
                    "no_repeat_ngram_banned_token_count": (
                        512 if controlled_decode else 0
                    ),
                    "decode_control_fallback_count": 0,
                },
                "promotion_gate": {
                    "diagnostic_boundary_reached": token_delta >= 8192,
                    "long_run_gate_reached": token_delta >= 131072,
                    "house_scale_gate_reached": token_delta >= 524288,
                    "promotes_runtime_claim": False,
                    "promotes_hot_path": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_gpu_kernel_report(path, *, kernel_name: str = RMSNORM_KERNEL_NAME) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": KERNEL_ARTIFACT_KIND,
                "surface": KERNEL_SURFACE,
                "kernel_name": kernel_name,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "parity_passed": True,
                "valid_shape_result_count": 2,
                "dtype_coverage": ["float16", "float32"],
                "benchmark_summary": {
                    "geometric_speedup_vs_torch": 1.25,
                },
                "promotion_gate": {
                    "kernel_parity_available": True,
                    "complete_runtime_impact_available": False,
                    "promotes_hot_path": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_generation_coherence_report(
    path,
    *,
    checkpoint_path: str = "reports/language_training_experiments/checkpoint.pt",
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": GENERATION_COHERENCE_ARTIFACT_KIND,
                "surface": GENERATION_COHERENCE_SURFACE,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "checkpoint_path": checkpoint_path,
                "prompt_suite": {
                    "review_kind": "automated_grounded_prompt_suite_not_human_review",
                },
                "summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "case_pass_rate": 1.0,
                    "mean_prefix_match_chars": 48.0,
                    "mean_prefix_match_fraction": 0.75,
                    "mean_printable_fraction": 1.0,
                    "mean_distinct_bigram_fraction": 0.8,
                    "next_character_match_rate": 1.0,
                },
                "promotion_gate": {
                    "generation_coherence_available": True,
                    "grounded_prompt_suite_available": True,
                    "human_review_available": False,
                    "promotes_generation_quality_claim": False,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_quality_replay_report(
    path,
    *,
    parent_checkpoint_path: str = "reports/language_training_experiments/parent.pt",
    child_checkpoint_path: str = "reports/language_training_experiments/checkpoint.pt",
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": QUALITY_REPLAY_ARTIFACT_KIND,
                "surface": QUALITY_REPLAY_SURFACE,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "parent_checkpoint_path": parent_checkpoint_path,
                "parent_checkpoint_sha256": "parent-hash",
                "child_checkpoint_path": child_checkpoint_path,
                "child_checkpoint_sha256": "child-hash",
                "checkpoint_lineage": {
                    "surface": "marulho_language_quality_replay_checkpoint_lineage.v1",
                    "parent_checkpoint_path": parent_checkpoint_path,
                    "child_checkpoint_path": child_checkpoint_path,
                    "parent_checkpoint_sha256": "parent-hash",
                    "child_checkpoint_sha256": "child-hash",
                    "candidate_count": 2,
                    "selected_candidate_id": "candidate-01",
                    "candidate_child_checkpoint_paths": [
                        "reports/language_training_experiments/rejected-child.pt",
                        child_checkpoint_path,
                    ],
                    "writes_child_checkpoint": True,
                    "mutates_parent_checkpoint": False,
                },
                "candidate_selection": {
                    "surface": "marulho_language_quality_replay_candidate_selection.v1",
                    "enabled": True,
                    "candidate_count": 2,
                    "selection_policy": (
                        "prefer_min_heldout_regression_then_heldout_gain_then_"
                        "trained_gain_then_learning_acceptance_then_update_throughput"
                    ),
                    "selected_candidate_id": "candidate-01",
                    "selected_candidate_index": 1,
                    "selected_child_checkpoint_path": child_checkpoint_path,
                    "selected_child_checkpoint_sha256": "child-hash",
                    "selected_selection_rank": [
                        -0.0,
                        0.0,
                        6.75,
                        -0.0,
                        1.0,
                        18.0,
                        0.0,
                        -0.0,
                        -0.0,
                        2796.957,
                    ],
                    "selected_selection_score": 18552.797,
                    "saves_child_checkpoint_per_candidate": True,
                    "runs_sustained_runtime_only_for_selected_child": True,
                    "mutates_parent_checkpoint": False,
                    "heldout_cases_used_for_replay_training": False,
                    "candidates": [
                        {
                            "candidate_id": "candidate-00",
                            "candidate_index": 0,
                            "selected": False,
                            "child_checkpoint_path": (
                                "reports/language_training_experiments/rejected-child.pt"
                            ),
                            "child_checkpoint_sha256": "rejected-hash",
                            "learning_status": "accepted_online_update",
                            "learning_config": {
                                "learning_rate": 0.0008,
                                "replay_loss_weight": 0.35,
                                "max_steps": 2,
                            },
                            "update_tokens_per_second": 3183.914,
                            "total_window_tokens_per_second": 1934.635,
                            "heldout_generation_coherence_delta": {
                                "surface": (
                                    "marulho_language_quality_replay_coherence_delta.v1"
                                ),
                                "regressed_prompt_count": 1,
                                "repaired_prompt_count": 0,
                                "mean_prefix_match_chars_delta": 2.5,
                            },
                        },
                        {
                            "candidate_id": "candidate-01",
                            "candidate_index": 1,
                            "selected": True,
                            "child_checkpoint_path": child_checkpoint_path,
                            "child_checkpoint_sha256": "child-hash",
                            "learning_status": "accepted_online_update",
                            "learning_config": {
                                "learning_rate": 0.0001,
                                "replay_loss_weight": 2.5,
                                "max_steps": 1,
                            },
                            "update_tokens_per_second": 2796.957,
                            "total_window_tokens_per_second": 1271.694,
                            "trained_generation_coherence_delta": {
                                "surface": (
                                    "marulho_language_quality_replay_coherence_delta.v1"
                                ),
                                "regressed_prompt_count": 0,
                                "repaired_prompt_count": 1,
                                "mean_prefix_match_chars_delta": 18.0,
                            },
                            "heldout_generation_coherence_delta": {
                                "surface": (
                                    "marulho_language_quality_replay_coherence_delta.v1"
                                ),
                                "regressed_prompt_count": 0,
                                "repaired_prompt_count": 0,
                                "mean_prefix_match_chars_delta": 6.75,
                            },
                        },
                    ],
                },
                "learning_evidence": {
                    "artifact_kind": "marulho_language_continual_learning_window",
                    "surface": "marulho_language_continual_learning_window.v1",
                    "owned_by_marulho": True,
                    "external_llm_used": False,
                    "loads_external_checkpoint": False,
                    "active_language_path": "marulho_lm_head",
                    "status": "accepted_online_update",
                    "learning_evidence": {
                        "tokens_per_second": 2796.957,
                        "total_window_tokens_per_second": 1271.694,
                        "update_token_count": 16384,
                        "new_domain_loss_delta": 0.058,
                        "old_domain_forgetting": -0.054,
                        "general_replay_retention_delta": -0.051,
                    },
                    "rollback_evidence": {
                        "restore_verified": True,
                        "rollback_required": False,
                    },
                    "promotion_gate": {
                        "eligible_for_online_learning_review": True,
                        "rollback_available": True,
                    },
                },
                "generation_coherence_after": {
                    "checkpoint_path": child_checkpoint_path,
                    "promotion_gate": {
                        "generation_coherence_available": True,
                        "grounded_prompt_suite_available": True,
                    },
                    "summary": {
                        "case_count": 4,
                        "passed_case_count": 4,
                        "case_pass_rate": 1.0,
                    },
                },
                "heldout_generation_coherence_after": {
                    "checkpoint_path": child_checkpoint_path,
                    "promotion_gate": {
                        "generation_coherence_available": True,
                        "grounded_prompt_suite_available": True,
                    },
                    "summary": {
                        "case_count": 4,
                        "passed_case_count": 4,
                        "case_pass_rate": 1.0,
                    },
                },
                "generation_coherence_delta": {
                    "surface": "marulho_language_quality_replay_coherence_delta.v1",
                    "regressed_prompt_count": 0,
                    "repaired_prompt_count": 1,
                    "mean_prefix_match_chars_delta": 18.0,
                },
                "heldout_generation_coherence_delta": {
                    "surface": "marulho_language_quality_replay_coherence_delta.v1",
                    "regressed_prompt_count": 0,
                    "repaired_prompt_count": 0,
                    "mean_prefix_match_chars_delta": 6.75,
                },
                "quality_generalization_review": {
                    "surface": "marulho_language_quality_replay_generalization_review.v1",
                    "trained_prompt_coherence_available": True,
                    "heldout_prompt_coherence_recorded": True,
                    "heldout_prompt_coherence_available": True,
                    "heldout_case_count": 4,
                    "heldout_passed_case_count": 4,
                    "heldout_case_pass_rate": 1.0,
                    "heldout_regressed_prompt_count": 0,
                    "heldout_repaired_prompt_count": 0,
                    "heldout_mean_prefix_match_chars_delta": 6.75,
                    "promotes_generation_quality_claim": False,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_memory_slot_runtime_impact_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": MEMORY_SLOT_RUNTIME_IMPACT_ARTIFACT_KIND,
                "surface": MEMORY_SLOT_RUNTIME_IMPACT_SURFACE,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "model_vocab_size": 524288,
                "batch": {
                    "tokens_per_forward": 1024,
                },
                "arms": {
                    "bounded_memory_slots_enabled": {
                        "candidate_slot_count": 8,
                        "active_slots_per_token": 2,
                        "candidate_slots_scored": 8192,
                        "runs_all_slots": False,
                    },
                    "all_slot_memory_scan_contrast": {
                        "candidate_slots_scored": 1048576,
                        "runs_all_slots": True,
                    },
                },
                "comparison": {
                    "control_tokens_per_second": 12783.3,
                    "bounded_tokens_per_second": 12171.6,
                    "bounded_vs_control_tokens_per_second_ratio": 0.952,
                    "all_slot_tokens_per_second": 10863.4,
                    "all_slot_vs_bounded_tokens_per_second_ratio": 0.893,
                    "bounded_memory_slot_nonzero_count": 512,
                    "bounded_memory_slot_gate_initial_value": 0.0,
                    "bounded_trainable_neutral_initialization": True,
                    "memory_gate_readback": False,
                },
                "promotion_gate": {
                    "complete_runtime_impact_available": True,
                    "bounded_memory_slots_enabled": True,
                    "bounded_avoids_all_slot_scan": True,
                    "neutral_initialization_parity": True,
                    "trainable_neutral_initialization": True,
                    "promotes_hot_path": False,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_memory_slot_architecture_cost_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_continual_learning_window",
                "surface": "marulho_language_continual_learning_window.v1",
                "experiment_surface": (
                    "marulho_language_continual_learning_experiment.v1"
                ),
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "status": "accepted_online_update",
                "model_vocab_size": 524288,
                "sampled_vocab_size": 1024,
                "learning_evidence": {
                    "update_token_count": 524288,
                    "tokens_per_second": 3753.246,
                    "total_window_tokens_per_second": 3436.735,
                    "new_domain_loss_delta": 7.0259,
                    "old_domain_forgetting": -7.0706,
                    "general_replay_retention_delta": -7.0833,
                    "memory_slots": {
                        "enabled": True,
                        "candidate_slots_scored": 4194304,
                        "candidate_id_source": (
                            "precomputed_batch_memory_candidate_ids"
                        ),
                        "memory_slot_retrieval_backend": (
                            "torch_autograd_bounded_memory_slots"
                        ),
                        "runs_all_slots": False,
                        "bounded_memory_slot_path": True,
                    },
                },
                "training_memory_slot_backend_summary": {
                    "surface": (
                        "marulho_language_continual_training_memory_slot_backend.v1"
                    ),
                    "training_window_stats_recorded": True,
                    "memory_slot_retrieval_backend": (
                        "torch_autograd_bounded_memory_slots"
                    ),
                    "triton_autograd_used": False,
                },
                "memory_slot_architecture_cost": {
                    "surface": MEMORY_SLOT_ARCHITECTURE_COST_SURFACE,
                    "status": "memory_slot_architecture_cost_measured",
                    "comparison_report": "reports/no-memory.json",
                    "comparison_is_no_memory_baseline": True,
                    "comparable_update_throughput": True,
                    "comparable_total_window_throughput": True,
                    "current_update_tokens_per_second": 3753.246,
                    "comparison_update_tokens_per_second": 3765.911,
                    "delta_vs_no_memory_update_percent": -0.336,
                    "current_total_window_tokens_per_second": 3436.735,
                    "comparison_total_window_tokens_per_second": 3451.048,
                    "delta_vs_no_memory_total_window_percent": -0.415,
                    "delta_vs_no_memory_new_domain_loss_delta": 0.00008,
                    "delta_vs_no_memory_old_domain_forgetting": -0.0055,
                    "delta_vs_no_memory_general_replay_retention_delta": -0.0077,
                    "delta_vs_no_memory_after_mean_source_prefix_match_chars": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_brain_installed_continual_learning_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": BRAIN_INSTALLED_CONTINUAL_LEARNING_ARTIFACT_KIND,
                "surface": BRAIN_INSTALLED_CONTINUAL_LEARNING_SURFACE,
                "status": "final",
                "report_status": "final",
                "runtime_owner": "MarulhoBrain",
                "active_language_path": "marulho_lm_head",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "service_owned_cognition": False,
                "status_read_mutation": False,
                "learning_summary": {
                    "surface": (
                        "marulho_brain_installed_continual_learning_summary.v1"
                    ),
                    "brain_surface": "marulho_brain_language_learning_window.v1",
                    "training_surface": (
                        "marulho_language_continual_learning_window.v1"
                    ),
                    "status": "accepted_online_update",
                    "trace_event": "language_learn",
                    "mutates_language_model_weights": True,
                    "update_token_count": 524288,
                    "tokens_per_second": 3079.877,
                    "total_window_tokens_per_second": 2810.819,
                    "new_domain_loss_delta": 4.7118,
                    "old_domain_forgetting": -4.0211,
                    "general_replay_retention_delta": -4.0014,
                    "final_parameter_delta_l2": 26.6627,
                    "device": "cuda:0",
                    "memory_slots": {
                        "enabled": True,
                        "candidate_slots_scored": 4194304,
                        "runs_all_slots": False,
                        "bounded_memory_slot_path": True,
                        "memory_slot_retrieval_backend": (
                            "torch_autograd_bounded_memory_slots"
                        ),
                    },
                },
                "pre_learning_brain_checkpoint": {
                    "surface": "marulho_brain_pre_learning_installed_checkpoint.v1",
                    "path": "reports/language_brain_continual_learning/pre.pt",
                    "restore_verified": True,
                },
                "learned_brain_checkpoint": {
                    "surface": "marulho_brain_post_learning_checkpoint.v1",
                    "path": "reports/language_brain_continual_learning/learned.pt",
                    "restore_verified": True,
                },
                "post_learning_sustained_window": {
                    "surface": (
                        "marulho_brain_post_learning_sustained_generation_summary.v1"
                    ),
                    "enabled": True,
                    "success": True,
                    "token_delta": 524288,
                    "tokens_per_second": 8132.276,
                    "backend": "torch_cuda_graph_burst_decode_controls",
                    "tracked_triton_kernel_used_names": [
                        "language_rmsnorm_triton",
                        "language_plif_triton",
                        "language_route_topk_triton",
                        "language_expert_dispatch_triton",
                        "language_memory_slots_triton",
                    ],
                    "tracked_triton_kernel_failure_count": 0,
                    "external_llm_used": False,
                    "service_owned_cognition": False,
                    "promotes_runtime_claim": False,
                },
                "promotion_gate": {
                    "surface": (
                        "marulho_language_brain_installed_continual_learning_gate.v1"
                    ),
                    "installed_reviewed_checkpoint": True,
                    "batch_tokenizer_matches_installed_runtime": True,
                    "pre_learning_brain_checkpoint_restore_verified": True,
                    "learning_runs_through_marulho_brain": True,
                    "language_learn_trace_recorded": True,
                    "records_actual_continual_learning": True,
                    "records_forgetting": True,
                    "records_replay_retention": True,
                    "records_update_throughput": True,
                    "records_total_window_throughput": True,
                    "house_scale_524288_update_tokens_reached": True,
                    "learned_brain_checkpoint_restore_verified": True,
                    "post_learning_sustained_524288_boundary_reached": True,
                    "status_read_mutation_absent": True,
                    "external_llm_absent": True,
                    "service_owned_cognition_absent": True,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_structural_plasticity_experiment_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": STRUCTURAL_PLASTICITY_EXPERIMENT_ARTIFACT_KIND,
                "surface": STRUCTURAL_PLASTICITY_EXPERIMENT_SURFACE,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "status": "completed_structural_plasticity_transactions",
                "model_vocab_size": 524288,
                "sampled_vocab_size": 1024,
                "transactions": [
                    {
                        "surface": (
                            "marulho_language_structural_plasticity_experiment_entry.v1"
                        ),
                        "proposal_kind": "memory_slot_expansion",
                        "transaction": {
                            "artifact_kind": (
                                STRUCTURAL_PLASTICITY_TRANSACTION_ARTIFACT_KIND
                            ),
                            "surface": STRUCTURAL_PLASTICITY_TRANSACTION_SURFACE,
                            "owned_by_marulho": True,
                            "external_llm_used": False,
                            "loads_external_checkpoint": False,
                            "active_language_path": "marulho_lm_head",
                            "status": "applied_structural_mutation",
                            "applied": True,
                            "mutates_runtime_state": True,
                            "operator_approved": True,
                            "checkpoint": {
                                "path": "reports/structural/baseline.pt",
                                "checkpoint_restore_verified": True,
                            },
                            "mutation": {
                                "proposal_kind": "memory_slot_expansion",
                                "source_expert_count": 16,
                                "target_expert_count": 16,
                                "source_route_candidate_count": 8,
                                "target_route_candidate_count": 8,
                                "source_memory_slot_count": 0,
                                "target_memory_slot_count": 1024,
                                "target_memory_slot_candidate_count": 8,
                                "target_active_memory_slot_count": 2,
                            },
                            "evaluation": {
                                "heldout_loss_delta": 0.0,
                            },
                            "rollback_evidence": {
                                "rollback_verified": True,
                            },
                            "promotion_gate": {
                                "checkpoint_backed": True,
                                "heldout_non_regression": True,
                                "eligible_for_reviewed_structural_promotion": True,
                                "promotes_runtime_claim": False,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_checkpoint_evolution_experiment_report(path) -> None:
    child_checkpoint = "reports/language_training_experiments/evolved-child.pt"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": CHECKPOINT_EVOLUTION_EXPERIMENT_ARTIFACT_KIND,
                "surface": CHECKPOINT_EVOLUTION_EXPERIMENT_SURFACE,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "status": "eligible_child_checkpoint_for_review",
                "checkpoint_lineage": {
                    "surface": "marulho_language_checkpoint_evolution_lineage.v1",
                    "lineage_id": "lineage-01",
                    "parent_checkpoint_path": (
                        "reports/language_training_experiments/parent.pt"
                    ),
                    "parent_checkpoint_sha256": "parent-sha",
                    "child_initial_checkpoint_path": (
                        "reports/language_training_experiments/child-initial.pt"
                    ),
                    "child_final_checkpoint_path": child_checkpoint,
                    "child_final_checkpoint_sha256": "child-sha",
                    "child_initial_matches_parent_state": True,
                    "child_final_matches_child_runtime": True,
                    "child_final_differs_from_parent_state": True,
                    "lineage_complete": True,
                },
                "runtime_evidence": {
                    "surface": (
                        "marulho_language_checkpoint_evolution_runtime_truth.v1"
                    ),
                    "child_training_device": "cuda:0",
                    "child_training_dense_adamw_backend": "fused",
                    "child_training_tokens_per_second": 3123.0,
                    "child_training_total_window_tokens_per_second": 2800.0,
                    "checkpoint_storage_device": "cpu",
                },
                "evolution_review": {
                    "surface": "marulho_language_checkpoint_evolution_review.v1",
                    "isolated_child_training": True,
                    "parent_kept_installed": True,
                    "child_update_token_count": 65536,
                    "child_optimizer_step_count": 64,
                    "structural_growth_attempted": True,
                    "structural_transaction_applied": True,
                    "long_run_evidence_required_for_promotion": True,
                },
                "experiment_review": {
                    "surface": (
                        "marulho_language_checkpoint_evolution_experiment_review.v1"
                    ),
                    "records_checkpoint_lineage": True,
                    "records_runtime_evidence": True,
                    "records_child_learning_update": True,
                },
                "split": {
                    "used_child_train_tokens": 65536,
                    "used_replay_tokens": 32768,
                },
                "promotion_gate": {
                    "checkpoint_evolution_evidence_available": True,
                    "eligible_for_parent_promotion_review": True,
                    "requires_operator_review": True,
                    "parent_runtime_unchanged": True,
                    "rollback_to_parent_verified": True,
                    "checkpoint_lineage_complete": True,
                    "child_checkpoint_available": True,
                    "long_run_evidence_required_for_parent_promotion": True,
                    "promotes_parent_promotion": False,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_language_runtime_benchmark_suite_writes_blocked_promotion_report(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=4,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    categories = {item["name"]: item for item in report["categories"]}

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert set(categories) == {
        "next_token_loss",
        "heldout_perplexity",
        "generation_coherence",
        "grounding_support",
        "continual_learning",
        "forgetting",
        "replay_recovery",
        "growth_prune_safety",
        "long_run_throughput",
        "active_compute",
        "memory_slot_runtime_impact",
        "memory_slot_architecture_cost",
        "gpu_kernel_correctness",
        "checkpoint_restore",
        "rollback",
        "service_contract",
        "scale_ladder",
    }
    assert categories["grounding_support"]["status"] == "pass"
    assert categories["grounding_support"]["evidence"]["source_term_coverage"] == 1.0
    assert categories["grounding_support"]["evidence"]["missing_required_terms"] == []
    assert categories["grounding_support"]["evidence"][
        "source_term_coverage_gate_passed"
    ] is True
    assert categories["gpu_kernel_correctness"]["status"] == "missing"
    assert categories["gpu_kernel_correctness"]["evidence"]["lm_triton_kernel_used"] is False
    assert categories["gpu_kernel_correctness"]["evidence"]["rmsnorm_triton_parity"] is False
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "plif_triton_forward_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "plif_triton_backward_surrogate_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "selective_scan_triton_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "local_eligibility_trace_update_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "route_vote_topk_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "block_sparse_expert_dispatch_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "sampled_vocab_cross_entropy_parity"
        ]
        is False
    )
    assert (
        categories["gpu_kernel_correctness"]["evidence"][
            "bounded_memory_slot_retrieval_parity"
        ]
        is False
    )
    assert "rmsnorm_triton_parity" in categories["gpu_kernel_correctness"]["missing_evidence"]
    assert (
        "plif_triton_forward_parity"
        in categories["gpu_kernel_correctness"]["missing_evidence"]
    )
    assert (
        "route_vote_topk_parity"
        in categories["gpu_kernel_correctness"]["missing_evidence"]
    )
    assert (
        "local_eligibility_trace_update_parity"
        in categories["gpu_kernel_correctness"]["missing_evidence"]
    )
    assert (
        "bounded_memory_slot_retrieval_parity"
        in categories["gpu_kernel_correctness"]["missing_evidence"]
    )
    assert categories["generation_coherence"]["status"] == "smoke_only"
    assert (
        "grounded_generation_coherence_report"
        in categories["generation_coherence"]["missing_evidence"]
    )
    assert categories["growth_prune_safety"]["evidence"]["growth_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["column_split_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["column_split_source_expert_count"] == 4
    assert categories["growth_prune_safety"]["evidence"]["column_split_target_expert_count"] == 5
    assert categories["growth_prune_safety"]["evidence"]["column_split_parent_child_pairs"] == [[1, 4]]
    assert categories["growth_prune_safety"]["evidence"]["prune_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["retire_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["retire_target_expert_count"] == 3
    assert categories["growth_prune_safety"]["evidence"]["retired_expert_ids"] == [3]
    assert categories["growth_prune_safety"]["evidence"]["merge_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["deep_sleep_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["deep_sleep_runs_all_columns"] is False
    assert categories["growth_prune_safety"]["evidence"]["route_bank_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["route_bank_source_candidate_count"] == 2
    assert categories["growth_prune_safety"]["evidence"]["route_bank_target_candidate_count"] == 4
    assert categories["growth_prune_safety"]["evidence"]["route_bank_runs_all_columns"] is False
    assert categories["growth_prune_safety"]["evidence"]["synapse_bundle_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["synapse_bundle_source_hidden_dim"] == 24
    assert categories["growth_prune_safety"]["evidence"]["synapse_bundle_target_hidden_dim"] == 32
    assert categories["growth_prune_safety"]["evidence"]["synapse_bundle_hidden_growth"] == 8
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_source_count"] == 0
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_target_count"] == 4
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_candidate_count"] == 2
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_active_count"] == 1
    assert categories["growth_prune_safety"]["evidence"]["memory_slot_runs_all_slots"] is False
    assert "column_split_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "prune_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "retire_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "merge_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "deep_sleep_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "route_bank_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "synapse_bundle_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "memory_slot_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert categories["growth_prune_safety"]["missing_evidence"] == []
    assert categories["growth_prune_safety"]["evidence"][
        "saved_structural_plasticity_evidence"
    ]["saved_structural_plasticity_evidence_available"] is False
    assert categories["long_run_throughput"]["status"] == "smoke_only"
    assert categories["long_run_throughput"]["evidence"]["smoke_token_delta"] == 4
    assert categories["long_run_throughput"]["evidence"][
        "diagnostic_boundary_reached"
    ] is False
    assert categories["long_run_throughput"]["evidence"]["long_run_gate_reached"] is False
    rollback = categories["rollback"]
    assert rollback["status"] == "pass"
    assert rollback["evidence"]["rollback_to_parent_verified"] is True
    assert rollback["evidence"]["parent_runtime_unchanged"] is True
    assert rollback["evidence"]["checkpoint_lineage_complete"] is True
    assert rollback["evidence"]["child_initial_matches_parent_state"] is True
    assert rollback["evidence"]["child_final_matches_child_runtime"] is True
    assert rollback["evidence"]["child_final_differs_from_parent_state"] is True
    assert rollback["evidence"]["parent_checkpoint_sha256"]
    assert rollback["evidence"]["child_final_checkpoint_sha256"]
    assert rollback["evidence"]["parent_kept_installed"] is True
    assert rollback["evidence"]["isolated_child_training"] is True
    assert rollback["evidence"]["child_update_token_count"] > 0
    assert rollback["evidence"]["operator_review_required"] is True
    assert rollback["evidence"]["long_run_evidence_required_for_promotion"] is True
    assert rollback["evidence"]["checkpoint_storage_device"] == "cpu"
    assert categories["service_contract"]["evidence"]["status_read_mutates_token_count"] is False
    assert categories["memory_slot_runtime_impact"]["status"] == "smoke_only"
    assert (
        categories["memory_slot_runtime_impact"]["evidence"][
            "memory_slot_runtime_impact_available"
        ]
        is False
    )
    assert categories["memory_slot_runtime_impact"]["missing_evidence"] == []
    assert categories["memory_slot_architecture_cost"]["status"] == "smoke_only"
    assert categories["memory_slot_architecture_cost"]["evidence"][
        "memory_slot_architecture_cost_available"
    ] is False
    assert categories["memory_slot_architecture_cost"]["missing_evidence"] == []
    assert categories["checkpoint_restore"]["status"] == "pass"
    assert report["promotion_gate"]["status"] == "blocked_missing_required_evidence"
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["requires_gpu_kernel_parity"] is True
    assert report["promotion_gate"]["requires_grounding_support"] is True
    assert report["promotion_gate"]["grounding_support_available"] is True
    assert report["promotion_gate"]["long_run_evidence_available"] is False
    assert report["promotion_gate"]["missing_required_category_names"] == [
        "generation_coherence",
        "long_run_throughput",
        "gpu_kernel_correctness",
    ]
    assert report["promotion_gate"]["requires_long_run_evidence"] is True
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "language-suite-grounding-support.json").exists()
    assert (tmp_path / "language-suite-sustained-smoke.json").exists()
    assert (tmp_path / "language-suite-scale-ladder.json").exists()
    assert (tmp_path / "language-suite-checkpoint.pt").exists()


def test_language_runtime_benchmark_suite_accepts_saved_lm_long_run_reports(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"
    diagnostic = tmp_path / "diagnostic-8192.json"
    long_gate = tmp_path / "long-gate-131072.json"
    controlled_house = tmp_path / "controlled-house-524288.json"
    rmsnorm_kernel = tmp_path / "rmsnorm-triton.json"
    plif_kernel = tmp_path / "plif-forward-triton.json"
    plif_surrogate_kernel = tmp_path / "plif-surrogate-triton.json"
    selective_scan_kernel = tmp_path / "selective-scan-triton.json"
    eligibility_trace_kernel = tmp_path / "eligibility-trace-triton.json"
    route_topk_kernel = tmp_path / "route-topk-triton.json"
    expert_dispatch_kernel = tmp_path / "expert-dispatch-triton.json"
    sampled_vocab_kernel = tmp_path / "sampled-vocab-ce-triton.json"
    memory_slot_kernel = tmp_path / "memory-slot-retrieval-triton.json"
    generation_coherence = tmp_path / "generation-coherence.json"
    quality_replay = tmp_path / "quality-replay.json"
    checkpoint_evolution = tmp_path / "checkpoint-evolution.json"
    memory_slot_runtime_impact = tmp_path / "memory-slot-runtime-impact.json"
    memory_slot_architecture_cost = tmp_path / "memory-slot-architecture-cost.json"
    brain_installed_continual_learning = (
        tmp_path / "brain-installed-continual-learning.json"
    )
    structural_plasticity = tmp_path / "structural-plasticity.json"
    _write_sustained_report(diagnostic, token_delta=8192)
    _write_sustained_report(long_gate, token_delta=131072)
    _write_sustained_report(
        controlled_house,
        token_delta=524288,
        controlled_decode=True,
    )
    _write_gpu_kernel_report(rmsnorm_kernel)
    _write_gpu_kernel_report(plif_kernel, kernel_name=PLIF_FORWARD_KERNEL_NAME)
    _write_gpu_kernel_report(
        plif_surrogate_kernel,
        kernel_name=PLIF_SURROGATE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        selective_scan_kernel,
        kernel_name=SELECTIVE_SCAN_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        eligibility_trace_kernel,
        kernel_name=ELIGIBILITY_TRACE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        route_topk_kernel,
        kernel_name=ROUTE_TOPK_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        expert_dispatch_kernel,
        kernel_name=EXPERT_DISPATCH_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        sampled_vocab_kernel,
        kernel_name=SAMPLED_VOCAB_CE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        memory_slot_kernel,
        kernel_name=MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
    )
    _write_generation_coherence_report(generation_coherence)
    _write_quality_replay_report(quality_replay)
    _write_checkpoint_evolution_experiment_report(checkpoint_evolution)
    _write_memory_slot_runtime_impact_report(memory_slot_runtime_impact)
    _write_memory_slot_architecture_cost_report(memory_slot_architecture_cost)
    _write_brain_installed_continual_learning_report(
        brain_installed_continual_learning
    )
    _write_structural_plasticity_experiment_report(structural_plasticity)

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=2,
        sustained_evidence_paths=(diagnostic, long_gate, controlled_house),
        brain_installed_continual_learning_evidence_paths=(
            brain_installed_continual_learning,
        ),
        memory_slot_runtime_impact_evidence_paths=(memory_slot_runtime_impact,),
        memory_slot_architecture_cost_evidence_paths=(
            memory_slot_architecture_cost,
        ),
        structural_plasticity_evidence_paths=(structural_plasticity,),
        gpu_kernel_evidence_paths=(
            rmsnorm_kernel,
            plif_kernel,
            plif_surrogate_kernel,
            selective_scan_kernel,
            eligibility_trace_kernel,
            route_topk_kernel,
            expert_dispatch_kernel,
            sampled_vocab_kernel,
            memory_slot_kernel,
        ),
        generation_coherence_evidence_paths=(generation_coherence,),
        quality_replay_evidence_paths=(quality_replay,),
        checkpoint_evolution_evidence_paths=(checkpoint_evolution,),
    )
    categories = {item["name"]: item for item in report["categories"]}
    long_run = categories["long_run_throughput"]
    gpu_kernel_category = categories["gpu_kernel_correctness"]
    generation_category = categories["generation_coherence"]
    memory_slot_category = categories["memory_slot_runtime_impact"]
    memory_slot_cost_category = categories["memory_slot_architecture_cost"]
    growth_prune_category = categories["growth_prune_safety"]
    rollback_category = categories["rollback"]
    continual_category = categories["continual_learning"]
    forgetting_category = categories["forgetting"]
    replay_category = categories["replay_recovery"]
    checkpoint_restore_category = categories["checkpoint_restore"]

    assert long_run["status"] == "pass"
    assert long_run["missing_evidence"] == []
    assert long_run["evidence"]["valid_report_count"] == 3
    assert long_run["evidence"]["diagnostic_boundary_reached"] is True
    assert long_run["evidence"]["long_run_gate_reached"] is True
    assert long_run["evidence"]["house_scale_gate_reached"] is True
    assert long_run["evidence"]["diagnostic_report"]["token_delta"] == 8192
    assert long_run["evidence"]["long_gate_report"]["token_delta"] == 131072
    assert long_run["evidence"]["house_scale_report"]["token_delta"] == 524288
    assert long_run["evidence"]["house_scale_report"]["checkpoint_path"] == (
        "reports/language_training_experiments/checkpoint.pt"
    )
    assert long_run["evidence"]["controlled_decode_report_count"] == 1
    assert long_run["evidence"]["controlled_decode_available"] is True
    assert long_run["evidence"]["controlled_decode_house_scale_gate_reached"] is True
    controlled_decode = long_run["evidence"]["controlled_decode_house_scale_report"][
        "generation_decode"
    ]
    assert controlled_decode["decode_controls_requested"] is True
    assert controlled_decode["decode_controls_backend"] == "torch_device_tensor"
    assert controlled_decode["decode_controls_cpu_token_copy"] is False
    assert controlled_decode["decode_controls_graph_compatible"] is True
    assert controlled_decode["cuda_graph_decode_controls_used"] is True
    assert controlled_decode["repetition_penalty_applied"] is True
    assert controlled_decode["repetition_penalty"] == 1.15
    assert controlled_decode["no_repeat_ngram_applied"] is True
    assert controlled_decode["no_repeat_ngram_size"] == 3
    assert controlled_decode["decode_control_fallback_count"] == 0
    assert long_run["evidence"]["promotes_runtime_claim"] is False
    assert long_run["evidence"]["promotes_hot_path"] is False
    assert gpu_kernel_category["status"] == "pass"
    assert generation_category["status"] == "pass"
    assert memory_slot_category["status"] == "pass"
    assert memory_slot_category["missing_evidence"] == []
    assert memory_slot_category["evidence"]["best_report"][
        "bounded_avoids_all_slot_scan"
    ] is True
    assert memory_slot_category["evidence"]["best_report"][
        "bounded_vs_control_tokens_per_second_ratio"
    ] == 0.952
    assert memory_slot_category["evidence"]["best_report"][
        "trainable_neutral_initialization"
    ] is True
    assert memory_slot_category["evidence"]["best_report"][
        "bounded_memory_slot_nonzero_count"
    ] == 512
    assert memory_slot_category["evidence"]["best_report"][
        "all_slot_runs_all_slots"
    ] is True
    assert memory_slot_category["evidence"]["required_for_runtime_promotion"] is False
    assert memory_slot_cost_category["status"] == "pass"
    assert memory_slot_cost_category["missing_evidence"] == []
    assert memory_slot_cost_category["evidence"][
        "memory_slot_architecture_cost_available"
    ] is True
    assert memory_slot_cost_category["evidence"]["best_report"][
        "delta_vs_no_memory_update_percent"
    ] == -0.336
    assert memory_slot_cost_category["evidence"]["best_report"][
        "delta_vs_no_memory_total_window_percent"
    ] == -0.415
    assert memory_slot_cost_category["evidence"]["best_report"][
        "candidate_slots_scored"
    ] == 4194304
    assert memory_slot_cost_category["evidence"]["best_report"][
        "runs_all_slots"
    ] is False
    assert memory_slot_cost_category["evidence"][
        "required_for_runtime_promotion"
    ] is False
    assert continual_category["status"] == "pass"
    brain_learning = continual_category["evidence"][
        "brain_installed_continual_learning_evidence"
    ]
    assert brain_learning["brain_installed_continual_learning_available"] is True
    assert brain_learning["valid_report_count"] == 1
    assert brain_learning["best_report"]["runtime_owner"] == "MarulhoBrain"
    assert brain_learning["best_report"]["brain_surface"] == (
        "marulho_brain_language_learning_window.v1"
    )
    assert brain_learning["best_report"]["update_token_count"] == 524288
    assert brain_learning["best_report"]["tokens_per_second"] == 3079.877
    assert brain_learning["best_report"]["total_window_tokens_per_second"] == 2810.819
    assert brain_learning["best_report"]["new_domain_loss_delta"] == 4.7118
    assert brain_learning["best_report"]["old_domain_forgetting"] == -4.0211
    assert (
        brain_learning["best_report"]["general_replay_retention_delta"]
        == -4.0014
    )
    assert brain_learning["best_report"]["memory_slot_candidate_slots_scored"] == 4194304
    assert brain_learning["best_report"]["memory_slot_runs_all_slots"] is False
    assert brain_learning["best_report"][
        "learned_brain_checkpoint_restore_verified"
    ] is True
    assert brain_learning["best_report"][
        "post_learning_sustained_524288_boundary_reached"
    ] is True
    assert (
        brain_learning["best_report"]["post_learning_sustained_tokens_per_second"]
        == 8132.276
    )
    assert continual_category["evidence"]["brain_installed_update_token_count"] == 524288
    assert (
        continual_category["evidence"]["brain_installed_tokens_per_second"]
        == 3079.877
    )
    assert forgetting_category["evidence"][
        "brain_installed_old_domain_forgetting"
    ] == -4.0211
    assert forgetting_category["evidence"][
        "brain_installed_forgetting_measured"
    ] is True
    assert replay_category["evidence"][
        "brain_installed_general_replay_retention_delta"
    ] == -4.0014
    assert replay_category["evidence"][
        "brain_installed_memory_slot_candidate_slots_scored"
    ] == 4194304
    assert replay_category["evidence"][
        "brain_installed_memory_slot_runs_all_slots"
    ] is False
    assert checkpoint_restore_category["evidence"][
        "brain_installed_learned_checkpoint_restore_verified"
    ] is True
    assert checkpoint_restore_category["evidence"][
        "brain_installed_learned_checkpoint_path"
    ] == "reports/language_brain_continual_learning/learned.pt"
    saved_structural = growth_prune_category["evidence"][
        "saved_structural_plasticity_evidence"
    ]
    assert saved_structural["saved_structural_plasticity_evidence_available"] is True
    assert saved_structural["valid_transaction_count"] == 1
    assert saved_structural["proposal_kinds"] == ["memory_slot_expansion"]
    assert saved_structural["transaction_summaries"][0][
        "target_memory_slot_count"
    ] == 1024
    assert rollback_category["status"] == "pass"
    saved_evolution = rollback_category["evidence"][
        "saved_checkpoint_evolution_evidence"
    ]
    assert saved_evolution["checkpoint_evolution_evidence_available"] is True
    assert saved_evolution["valid_report_count"] == 1
    assert saved_evolution["best_report"]["child_training_device"] == "cuda:0"
    assert saved_evolution["best_report"][
        "child_training_dense_adamw_backend"
    ] == "fused"
    assert saved_evolution["best_report"]["child_update_token_count"] == 65536
    assert saved_evolution["best_report"][
        "child_training_tokens_per_second"
    ] == 3123.0
    assert saved_evolution["best_report"][
        "long_run_evidence_required_for_parent_promotion"
    ] is True
    assert generation_category["missing_evidence"] == []
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_long_run_available"
    ] is True
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_house_scale_available"
    ] is True
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_controlled_decode_house_scale_available"
    ] is True
    assert (
        generation_category["evidence"]["best_report"]["review_kind"]
        == "automated_grounded_prompt_suite_not_human_review"
    )
    quality_replay_evidence = generation_category["evidence"][
        "quality_replay_evidence"
    ]
    assert quality_replay_evidence["quality_replay_available"] is True
    assert quality_replay_evidence["valid_report_count"] == 1
    assert quality_replay_evidence["best_report"]["candidate_count"] == 2
    assert (
        quality_replay_evidence["best_report"]["selected_candidate_id"]
        == "candidate-01"
    )
    assert (
        quality_replay_evidence["best_report"]["selected_child_checkpoint_path"]
        == "reports/language_training_experiments/checkpoint.pt"
    )
    assert (
        quality_replay_evidence["best_report"]["selected_update_tokens_per_second"]
        == 2796.957
    )
    assert (
        quality_replay_evidence["best_report"]["trained_repaired_prompt_count"]
        == 1
    )
    assert (
        quality_replay_evidence["best_report"]["heldout_regressed_prompt_count"]
        == 0
    )
    assert (
        quality_replay_evidence["best_report"][
            "heldout_mean_prefix_match_chars_delta"
        ]
        == 6.75
    )
    quality_replay_alignment = generation_category["evidence"][
        "quality_replay_long_run_alignment"
    ]
    assert quality_replay_alignment["same_child_long_run_available"] is True
    assert quality_replay_alignment["same_child_house_scale_available"] is True
    assert (
        quality_replay_alignment[
            "same_child_controlled_decode_house_scale_available"
        ]
        is True
    )
    assert (
        quality_replay_alignment["controlled_decode_house_scale_required"] is True
    )
    assert quality_replay_alignment["matching_report_count"] == 3
    assert gpu_kernel_category["evidence"]["lm_triton_kernel_used"] is True
    assert gpu_kernel_category["evidence"]["rmsnorm_triton_parity"] is True
    assert gpu_kernel_category["evidence"]["plif_triton_forward_parity"] is True
    assert (
        gpu_kernel_category["evidence"]["plif_triton_backward_surrogate_parity"]
        is True
    )
    assert gpu_kernel_category["evidence"]["selective_scan_triton_parity"] is True
    assert (
        gpu_kernel_category["evidence"]["local_eligibility_trace_update_parity"]
        is True
    )
    assert gpu_kernel_category["evidence"]["route_vote_topk_parity"] is True
    assert (
        gpu_kernel_category["evidence"]["block_sparse_expert_dispatch_parity"]
        is True
    )
    assert (
        gpu_kernel_category["evidence"]["sampled_vocab_cross_entropy_parity"]
        is True
    )
    assert (
        gpu_kernel_category["evidence"]["bounded_memory_slot_retrieval_parity"]
        is True
    )
    assert gpu_kernel_category["evidence"]["covered_kernel_names"] == [
        EXPERT_DISPATCH_KERNEL_NAME,
        ELIGIBILITY_TRACE_KERNEL_NAME,
        MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
        PLIF_FORWARD_KERNEL_NAME,
        PLIF_SURROGATE_KERNEL_NAME,
        RMSNORM_KERNEL_NAME,
        ROUTE_TOPK_KERNEL_NAME,
        SAMPLED_VOCAB_CE_KERNEL_NAME,
        SELECTIVE_SCAN_KERNEL_NAME,
    ]
    assert "rmsnorm_triton_parity" not in gpu_kernel_category["missing_evidence"]
    assert "plif_triton_forward_parity" not in gpu_kernel_category["missing_evidence"]
    assert (
        "plif_triton_backward_surrogate_parity"
        not in gpu_kernel_category["missing_evidence"]
    )
    assert "selective_scan_triton_parity" not in gpu_kernel_category["missing_evidence"]
    assert (
        "local_eligibility_trace_update_parity"
        not in gpu_kernel_category["missing_evidence"]
    )
    assert "route_vote_topk_parity" not in gpu_kernel_category["missing_evidence"]
    assert (
        "block_sparse_expert_dispatch_parity"
        not in gpu_kernel_category["missing_evidence"]
    )
    assert "sampled_vocab_cross_entropy_parity" not in gpu_kernel_category[
        "missing_evidence"
    ]
    assert "bounded_memory_slot_retrieval_parity" not in gpu_kernel_category[
        "missing_evidence"
    ]
    assert report["promotion_gate"]["long_run_evidence_available"] is True
    assert report["promotion_gate"]["generation_coherence_available"] is True
    assert report["promotion_gate"]["quality_replay_evidence_available"] is True
    assert report["promotion_gate"][
        "brain_installed_continual_learning_evidence_available"
    ] is True
    assert report["promotion_gate"][
        "checkpoint_evolution_evidence_available"
    ] is True
    assert (
        report["promotion_gate"][
            "controlled_decode_house_scale_evidence_available"
        ]
        is True
    )
    assert (
        report["promotion_gate"][
            "generation_controlled_decode_house_scale_aligned"
        ]
        is True
    )
    assert (
        report["promotion_gate"][
            "quality_replay_controlled_decode_house_scale_aligned"
        ]
        is True
    )
    assert report["promotion_gate"]["missing_required_category_names"] == []
    assert report["promotion_gate"]["status"] == "ready_for_review"


def test_language_runtime_benchmark_suite_blocks_mixed_checkpoint_quality_and_speed(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"
    diagnostic = tmp_path / "diagnostic-8192.json"
    long_gate = tmp_path / "long-gate-131072.json"
    generation_coherence = tmp_path / "generation-coherence.json"
    rmsnorm_kernel = tmp_path / "rmsnorm-triton.json"
    plif_kernel = tmp_path / "plif-forward-triton.json"
    plif_surrogate_kernel = tmp_path / "plif-surrogate-triton.json"
    selective_scan_kernel = tmp_path / "selective-scan-triton.json"
    eligibility_trace_kernel = tmp_path / "eligibility-trace-triton.json"
    route_topk_kernel = tmp_path / "route-topk-triton.json"
    expert_dispatch_kernel = tmp_path / "expert-dispatch-triton.json"
    sampled_vocab_kernel = tmp_path / "sampled-vocab-ce-triton.json"
    memory_slot_kernel = tmp_path / "memory-slot-retrieval-triton.json"

    _write_sustained_report(
        diagnostic,
        token_delta=8192,
        checkpoint_path="reports/language_training_experiments/fast-checkpoint.pt",
    )
    _write_sustained_report(
        long_gate,
        token_delta=131072,
        checkpoint_path="reports/language_training_experiments/fast-checkpoint.pt",
    )
    _write_generation_coherence_report(
        generation_coherence,
        checkpoint_path="reports/language_training_experiments/quality-checkpoint.pt",
    )
    _write_gpu_kernel_report(rmsnorm_kernel)
    _write_gpu_kernel_report(plif_kernel, kernel_name=PLIF_FORWARD_KERNEL_NAME)
    _write_gpu_kernel_report(
        plif_surrogate_kernel,
        kernel_name=PLIF_SURROGATE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        selective_scan_kernel,
        kernel_name=SELECTIVE_SCAN_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        eligibility_trace_kernel,
        kernel_name=ELIGIBILITY_TRACE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        route_topk_kernel,
        kernel_name=ROUTE_TOPK_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        expert_dispatch_kernel,
        kernel_name=EXPERT_DISPATCH_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        sampled_vocab_kernel,
        kernel_name=SAMPLED_VOCAB_CE_KERNEL_NAME,
    )
    _write_gpu_kernel_report(
        memory_slot_kernel,
        kernel_name=MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
    )

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=2,
        sustained_evidence_paths=(diagnostic, long_gate),
        gpu_kernel_evidence_paths=(
            rmsnorm_kernel,
            plif_kernel,
            plif_surrogate_kernel,
            selective_scan_kernel,
            eligibility_trace_kernel,
            route_topk_kernel,
            expert_dispatch_kernel,
            sampled_vocab_kernel,
            memory_slot_kernel,
        ),
        generation_coherence_evidence_paths=(generation_coherence,),
    )
    categories = {item["name"]: item for item in report["categories"]}
    generation_category = categories["generation_coherence"]

    assert categories["long_run_throughput"]["status"] == "pass"
    assert generation_category["status"] == "smoke_only"
    assert "same_checkpoint_generation_coherence_long_run" in generation_category[
        "missing_evidence"
    ]
    assert generation_category["evidence"]["generation_coherence_available"] is True
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_long_run_available"
    ] is False
    assert generation_category["evidence"]["long_run_alignment"][
        "generation_checkpoint_path"
    ] == "reports/language_training_experiments/quality-checkpoint.pt"
    assert report["promotion_gate"]["missing_required_category_names"] == [
        "generation_coherence"
    ]
    assert report["promotion_gate"]["status"] == "blocked_missing_required_evidence"


def test_language_runtime_benchmark_suite_blocks_mixed_controlled_decode_checkpoint(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"
    diagnostic = tmp_path / "diagnostic-8192.json"
    long_gate = tmp_path / "long-gate-131072.json"
    controlled_house = tmp_path / "controlled-house-524288.json"
    generation_coherence = tmp_path / "generation-coherence.json"
    quality_replay = tmp_path / "quality-replay.json"
    kernel_reports = (
        (tmp_path / "rmsnorm-triton.json", RMSNORM_KERNEL_NAME),
        (tmp_path / "plif-forward-triton.json", PLIF_FORWARD_KERNEL_NAME),
        (tmp_path / "plif-surrogate-triton.json", PLIF_SURROGATE_KERNEL_NAME),
        (tmp_path / "selective-scan-triton.json", SELECTIVE_SCAN_KERNEL_NAME),
        (tmp_path / "eligibility-trace-triton.json", ELIGIBILITY_TRACE_KERNEL_NAME),
        (tmp_path / "route-topk-triton.json", ROUTE_TOPK_KERNEL_NAME),
        (tmp_path / "expert-dispatch-triton.json", EXPERT_DISPATCH_KERNEL_NAME),
        (tmp_path / "sampled-vocab-ce-triton.json", SAMPLED_VOCAB_CE_KERNEL_NAME),
        (
            tmp_path / "memory-slot-retrieval-triton.json",
            MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
        ),
    )

    child_checkpoint = "reports/language_training_experiments/quality-child.pt"
    fast_checkpoint = "reports/language_training_experiments/fast-controlled.pt"
    _write_sustained_report(
        diagnostic,
        token_delta=8192,
        checkpoint_path=child_checkpoint,
    )
    _write_sustained_report(
        long_gate,
        token_delta=131072,
        checkpoint_path=child_checkpoint,
    )
    _write_sustained_report(
        controlled_house,
        token_delta=524288,
        controlled_decode=True,
        checkpoint_path=fast_checkpoint,
    )
    _write_generation_coherence_report(
        generation_coherence,
        checkpoint_path=child_checkpoint,
    )
    _write_quality_replay_report(
        quality_replay,
        child_checkpoint_path=child_checkpoint,
    )
    for kernel_path, kernel_name in kernel_reports:
        _write_gpu_kernel_report(kernel_path, kernel_name=kernel_name)

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=2,
        sustained_evidence_paths=(diagnostic, long_gate, controlled_house),
        gpu_kernel_evidence_paths=tuple(
            kernel_path for kernel_path, _kernel_name in kernel_reports
        ),
        generation_coherence_evidence_paths=(generation_coherence,),
        quality_replay_evidence_paths=(quality_replay,),
    )
    categories = {item["name"]: item for item in report["categories"]}
    generation_category = categories["generation_coherence"]
    quality_replay_alignment = generation_category["evidence"][
        "quality_replay_long_run_alignment"
    ]

    assert categories["long_run_throughput"]["status"] == "pass"
    assert generation_category["status"] == "smoke_only"
    assert (
        "same_checkpoint_generation_coherence_controlled_decode_house_scale"
        in generation_category["missing_evidence"]
    )
    assert (
        "same_child_quality_replay_controlled_decode_house_scale"
        in generation_category["missing_evidence"]
    )
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_long_run_available"
    ] is True
    assert generation_category["evidence"]["long_run_alignment"][
        "same_checkpoint_controlled_decode_house_scale_available"
    ] is False
    assert (
        quality_replay_alignment[
            "same_child_controlled_decode_house_scale_available"
        ]
        is False
    )
    assert quality_replay_alignment["controlled_decode_house_scale_required"] is True
    assert (
        report["promotion_gate"][
            "controlled_decode_house_scale_evidence_available"
        ]
        is True
    )
    assert (
        report["promotion_gate"][
            "generation_controlled_decode_house_scale_aligned"
        ]
        is False
    )
    assert (
        report["promotion_gate"][
            "quality_replay_controlled_decode_house_scale_aligned"
        ]
        is False
    )
    assert report["promotion_gate"]["missing_required_category_names"] == [
        "generation_coherence"
    ]
    assert report["promotion_gate"]["status"] == "blocked_missing_required_evidence"
