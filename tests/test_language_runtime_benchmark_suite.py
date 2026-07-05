from __future__ import annotations

import json

from marulho.evaluation.language_runtime_benchmark_suite import (
    ELIGIBILITY_TRACE_KERNEL_NAME,
    EXPERT_DISPATCH_KERNEL_NAME,
    GENERATION_COHERENCE_ARTIFACT_KIND,
    GENERATION_COHERENCE_SURFACE,
    KERNEL_ARTIFACT_KIND,
    KERNEL_SURFACE,
    MEMORY_SLOT_RUNTIME_IMPACT_ARTIFACT_KIND,
    MEMORY_SLOT_RUNTIME_IMPACT_SURFACE,
    PLIF_FORWARD_KERNEL_NAME,
    PLIF_SURROGATE_KERNEL_NAME,
    RMSNORM_KERNEL_NAME,
    ROUTE_TOPK_KERNEL_NAME,
    SAMPLED_VOCAB_CE_KERNEL_NAME,
    SELECTIVE_SCAN_KERNEL_NAME,
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
    assert categories["long_run_throughput"]["status"] == "smoke_only"
    assert categories["long_run_throughput"]["evidence"]["smoke_token_delta"] == 4
    assert categories["long_run_throughput"]["evidence"][
        "diagnostic_boundary_reached"
    ] is False
    assert categories["long_run_throughput"]["evidence"]["long_run_gate_reached"] is False
    assert categories["service_contract"]["evidence"]["status_read_mutates_token_count"] is False
    assert categories["memory_slot_runtime_impact"]["status"] == "smoke_only"
    assert (
        categories["memory_slot_runtime_impact"]["evidence"][
            "memory_slot_runtime_impact_available"
        ]
        is False
    )
    assert categories["memory_slot_runtime_impact"]["missing_evidence"] == []
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
    generation_coherence = tmp_path / "generation-coherence.json"
    memory_slot_runtime_impact = tmp_path / "memory-slot-runtime-impact.json"
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
    _write_generation_coherence_report(generation_coherence)
    _write_memory_slot_runtime_impact_report(memory_slot_runtime_impact)

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=2,
        sustained_evidence_paths=(diagnostic, long_gate, controlled_house),
        memory_slot_runtime_impact_evidence_paths=(memory_slot_runtime_impact,),
        gpu_kernel_evidence_paths=(
            rmsnorm_kernel,
            plif_kernel,
            plif_surrogate_kernel,
            selective_scan_kernel,
            eligibility_trace_kernel,
            route_topk_kernel,
            expert_dispatch_kernel,
            sampled_vocab_kernel,
        ),
        generation_coherence_evidence_paths=(generation_coherence,),
    )
    categories = {item["name"]: item for item in report["categories"]}
    long_run = categories["long_run_throughput"]
    gpu_kernel_category = categories["gpu_kernel_correctness"]
    generation_category = categories["generation_coherence"]
    memory_slot_category = categories["memory_slot_runtime_impact"]

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
    assert gpu_kernel_category["evidence"]["covered_kernel_names"] == [
        EXPERT_DISPATCH_KERNEL_NAME,
        ELIGIBILITY_TRACE_KERNEL_NAME,
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
    assert report["promotion_gate"]["long_run_evidence_available"] is True
    assert report["promotion_gate"]["generation_coherence_available"] is True
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
