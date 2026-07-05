from __future__ import annotations

import json

import torch

from marulho.evaluation.language_training_experiment import (
    SURFACE,
    LanguageTrainingExperimentConfig,
    run_language_training_experiment,
)


def test_language_training_experiment_trains_generates_and_streams(tmp_path) -> None:
    output = tmp_path / "language-experiment.json"

    report = run_language_training_experiment(
        output_path=output,
        prompts=("MARULHO", "replay evidence"),
        config=LanguageTrainingExperimentConfig(
            embedding_dim=10,
            state_dim=16,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_train_batches=4,
            train_epochs=1,
            generation_tokens=4,
            sustained_target_tokens=3,
            sustained_tick_tokens=2,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["training"]["token_count"] > 0
    assert report["training"]["tokens_per_second"] > 0.0
    assert report["training"]["batch_size"] == 2
    assert report["training"]["max_tokens_per_optimizer_step"] > 12
    assert report["training"]["loss_start"] is not None
    assert report["training"]["loss_end"] is not None
    assert report["training"]["metric_readback_mode"] == (
        "deferred_gpu_scalar_aggregation"
    )
    assert report["training"]["per_batch_metric_cpu_sync"] is False
    assert report["training"]["training_stage_profile"]["enabled"] is False
    assert report["training"]["optimizer_step_count"] == 4
    assert report["training"]["gradient_clip_mode"] == (
        "sparse_aware_device_norm_every_step"
    )
    assert report["training"]["gradient_clip_interval"] == 1
    assert report["training"]["gradient_clip_applied_step_count"] == 4
    assert report["training"]["gradient_clip_skipped_step_count"] == 0
    assert report["training"]["gradient_norm_observed_step_count"] == 4
    assert report["cuda_math_policy"]["applied"] is False
    assert report["cuda_math_policy"]["requested_matmul_allow_tf32"] is True
    assert report["training"]["cuda_math_policy"] == report["cuda_math_policy"]
    assert report["training"]["recurrent_gradient_horizon"] == 0
    assert report["training"]["truncated_recurrent_bptt"] is False
    assert report["training"]["truncated_bptt_boundary_count_per_batch"] == 0
    assert report["training"]["state_block_projection_mode"] == (
        "batched_token_and_state_output_projection_recurrent_loop"
    )
    assert report["training"]["state_output_projection_batched"] is True
    assert report["training"]["expert_dispatch_backend"] == (
        "torch_selected_expert_batched_matmul_dispatch"
    )
    assert report["training"]["expert_training_dispatch_batched_matmul"] is True
    assert report["training"]["loss_record_count"] == 4
    assert report["training"]["cuda_synchronized_before_timing_start"] is False
    assert report["training"]["cuda_synchronized_before_timing_stop"] is False
    assert report["training"]["language_plif_triton"]["triton_available"] in {
        True,
        False,
    }
    assert report["training"]["plif_surrogate_triton_used"] is False
    assert report["eval_before"]["heldout_perplexity"] > 0.0
    assert report["eval_after"]["heldout_perplexity"] > 0.0
    assert len(report["generation_after"]) == 2
    assert report["generation_after"][0]["external_llm_used"] is False
    assert report["generation_after"][0]["new_token_count"] >= 1
    assert report["generation_after"][0]["quality_probe"]["printable_fraction"] >= 0.0
    assert "source_continuation" in report["generation_after"][0]["quality_probe"]
    assert report["generation_quality_after"]["generation_count"] == 2
    assert report["generation_quality_after"]["review_kind"] == (
        "source_continuation_probe_not_human_quality_review"
    )
    assert "mean_source_prefix_match_chars_delta" in report["generation_quality_delta"]
    assert report["sustained_summary"]["success"] is True
    assert report["sustained_summary"]["token_delta"] == 3
    assert report["experiment_review"]["fast_mutable_experiment"] is True
    assert report["experiment_review"]["records_actual_training"] is True
    assert report["experiment_review"]["records_actual_generation"] is True
    assert report["experiment_review"]["records_generation_quality_probe"] is True
    assert report["experiment_review"]["promotes_runtime_claim"] is False
    assert (tmp_path / "language-experiment-checkpoint.pt").exists()
    assert (tmp_path / "language-experiment-sustained.json").exists()
    assert (tmp_path / "README.md").exists()


def test_language_training_experiment_supports_sampled_padded_vocab(tmp_path) -> None:
    output = tmp_path / "language-sampled-padded.json"

    report = run_language_training_experiment(
        output_path=output,
        prompts=("MARULHO",),
        config=LanguageTrainingExperimentConfig(
            model_vocab_size=384,
            sampled_vocab_size=32,
            sparse_vocab_optimizer=True,
            embedding_dim=10,
            state_dim=16,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
            recurrent_gradient_horizon=4,
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_train_batches=3,
            train_epochs=1,
            gradient_clip_interval=2,
            generation_tokens=4,
            sustained_target_tokens=3,
            sustained_tick_tokens=2,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            profile_training_stages=True,
            device="cpu",
        ),
    )

    assert report["model_vocab_size"] == 384
    assert report["tokenizer_vocab_size"] < report["model_vocab_size"]
    assert report["generation_vocab_size"] == report["tokenizer_vocab_size"]
    assert report["generation_decode"]["full_model_vocab_logits_materialized"] is False
    assert report["training"]["optimizer_policy"] == (
        "AdamW_dense_core_plus_SparseAdam_vocab_rows"
    )
    assert report["training"]["loss_kind"] == "sampled_adaptive_vocab_cross_entropy"
    assert report["training"]["sampled_vocab_training"] is True
    assert report["training"]["optimizer_step_count"] == 3
    assert report["training"]["gradient_clip_mode"] == (
        "sparse_aware_device_norm_every_n_steps"
    )
    assert report["training"]["gradient_clip_interval"] == 2
    assert report["training"]["gradient_clip_applied_step_count"] == 1
    assert report["training"]["gradient_clip_skipped_step_count"] == 2
    assert report["training"]["gradient_norm_observed_step_count"] == 1
    assert report["config"]["cuda_allow_tf32"] is True
    assert report["config"]["cuda_float32_matmul_precision"] == "high"
    assert report["experiment_review"]["records_cuda_math_policy"] is False
    assert report["training"]["recurrent_gradient_horizon"] == 4
    assert report["training"]["truncated_recurrent_bptt"] is True
    assert report["training"]["gradient_horizon_policy"] == (
        "bounded_recurrent_state_detach"
    )
    assert report["training"]["state_block_gradient_horizon_policy"] == (
        "bounded_recurrent_state_detach"
    )
    assert report["training"]["state_block_projection_mode"] == (
        "batched_token_and_state_output_projection_recurrent_loop"
    )
    assert report["training"]["state_output_projection_batched"] is True
    assert report["training"]["expert_dispatch_backend"] == (
        "torch_selected_expert_batched_matmul_dispatch"
    )
    assert report["training"]["expert_training_dispatch_batched_matmul"] is True
    assert report["training"]["truncated_bptt_boundary_count_per_batch"] == 2
    assert report["model_config"]["recurrent_gradient_horizon"] == 4
    assert report["training"]["full_vocab_logits_materialized"] is False
    assert report["training"]["loss_evidence"]["lm_head_weight_gradient_sparse"] is True
    assert report["training"]["loss_evidence"]["token_embedding_gradient_sparse"] is True
    assert report["training"]["sampled_vocab_precompute"]["enabled"] is True
    assert report["training"]["sampled_vocab_precompute"]["batch_count"] == 3
    assert report["training"]["sampled_vocab_precompute"][
        "hot_update_window_precomputed"
    ] is True
    assert report["training"]["loss_evidence"]["sampled_vocab_id_source"] == (
        "precomputed_batch_sampled_vocab_ids"
    )
    assert report["training"]["loss_evidence"]["sampled_target_position_source"] == (
        "precomputed_batch_target_positions"
    )
    assert report["training"]["loss_evidence"]["precomputed_sampled_vocab_used"] is True
    assert report["training"]["loss_evidence"]["precomputed_target_positions_used"] is True
    stage_profile = report["training"]["training_stage_profile"]
    assert stage_profile["enabled"] is True
    assert stage_profile["measurement"] == "host_perf_counter"
    assert stage_profile["per_stage"]["forward_loss"]["count"] == 3
    assert stage_profile["per_stage"]["backward"]["mean_ms_per_token"] >= 0.0
    assert stage_profile["per_stage"]["optimizer_step"]["mean_ms_per_token"] >= 0.0
    assert stage_profile["top_stage_mean_ms_per_token"]
    assert report["generation_after"][0]["generation_decode"][
        "full_model_vocab_logits_materialized"
    ] is False
    assert report["sustained_summary"]["generation_vocab_size"] == (
        report["tokenizer_vocab_size"]
    )
    assert report["experiment_review"]["records_sampled_vocab_training"] is True
    assert report["experiment_review"]["records_padded_vocab_decode_policy"] is True


def test_language_training_experiment_reports_memory_slot_training_and_sustain(
    tmp_path,
) -> None:
    output = tmp_path / "language-memory-slots.json"

    report = run_language_training_experiment(
        output_path=output,
        prompts=("MARULHO",),
        config=LanguageTrainingExperimentConfig(
            model_vocab_size=384,
            sampled_vocab_size=32,
            sparse_vocab_optimizer=True,
            embedding_dim=10,
            state_dim=16,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
            recurrent_gradient_horizon=4,
            memory_slot_count=4,
            memory_slot_candidate_count=2,
            active_memory_slot_count=1,
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_train_batches=2,
            train_epochs=1,
            generation_tokens=4,
            sustained_target_tokens=3,
            sustained_tick_tokens=2,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            device="cpu",
        ),
    )

    assert report["model_config"]["memory_slot_count"] == 4
    assert report["model_config"]["memory_slot_candidate_count"] == 2
    assert report["model_config"]["active_memory_slot_count"] == 1
    assert report["training"]["sampled_vocab_training"] is True
    assert report["training"]["full_vocab_logits_materialized"] is False
    assert report["training"]["memory_enabled"] is True
    assert report["training"]["memory_total_slots"] == 4
    assert report["training"]["memory_candidate_slot_count"] == 2
    assert report["training"]["memory_active_slots_per_token"] == 1
    assert report["training"]["memory_candidate_slots_scored"] > 0
    assert report["training"]["memory_runs_all_slots"] is False
    assert report["training"]["memory_candidate_id_source"] == (
        "precomputed_batch_memory_candidate_ids"
    )
    memory_precompute = report["training"]["memory_candidate_precompute"]
    assert memory_precompute["surface"] == (
        "marulho_language_memory_candidate_batch_precompute.v1"
    )
    assert memory_precompute["enabled"] is True
    assert memory_precompute["hot_update_window_precomputed"] is True
    assert memory_precompute["candidate_id_source"] == (
        "precomputed_batch_memory_candidate_ids"
    )
    assert memory_precompute["batch_count"] == 2
    assert report["training"]["sampled_vocab_precompute"][
        "memory_candidate_precompute"
    ] == memory_precompute
    assert report["training"]["memory_gate_readback"] is False
    memory_slots = report["sustained_summary"]["memory_slots"]
    assert memory_slots["surface"] == "marulho_language_sustained_memory_slots.v1"
    assert memory_slots["enabled"] is True
    assert memory_slots["total_slots"] == 4
    assert memory_slots["candidate_slot_count"] == 2
    assert memory_slots["active_slots_per_token"] == 1
    assert memory_slots["runs_all_slots"] is False
    assert memory_slots["candidate_id_source"] == "token_hash_memory_slot_bank"
    assert memory_slots["memory_gate_readback"] is False
    assert report["experiment_review"]["records_memory_slot_path"] is True
    assert report["experiment_review"]["records_bounded_memory_slot_path"] is True


def test_language_training_experiment_restores_cuda_math_policy(tmp_path) -> None:
    before_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    before_cudnn = bool(torch.backends.cudnn.allow_tf32)
    before_precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else "unavailable"
    )

    report = run_language_training_experiment(
        output_path=tmp_path / "language-tf32-policy.json",
        prompts=("MARULHO",),
        config=LanguageTrainingExperimentConfig(
            embedding_dim=8,
            state_dim=12,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=1,
            expert_hidden_dim=16,
            sequence_length=8,
            stride=4,
            batch_size=2,
            max_train_batches=2,
            train_epochs=1,
            generation_tokens=2,
            sustained_target_tokens=2,
            sustained_tick_tokens=1,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            cuda_allow_tf32=False,
            cuda_float32_matmul_precision="highest",
            device="cpu",
        ),
    )

    assert report["cuda_math_policy"]["requested_matmul_allow_tf32"] is False
    assert report["cuda_math_policy"]["requested_float32_matmul_precision"] == "highest"
    assert report["cuda_math_policy"]["applied"] is False
    assert bool(torch.backends.cuda.matmul.allow_tf32) is before_matmul
    assert bool(torch.backends.cudnn.allow_tf32) is before_cudnn
    if hasattr(torch, "get_float32_matmul_precision"):
        assert torch.get_float32_matmul_precision() == before_precision
