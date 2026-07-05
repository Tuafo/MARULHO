from __future__ import annotations

from marulho.evaluation.language_memory_slot_training_impact import (
    MemorySlotTrainingImpactConfig,
    run_language_memory_slot_training_impact,
)


def test_language_memory_slot_training_impact_reports_full_step(tmp_path) -> None:
    output = tmp_path / "memory-slot-training-impact.json"

    report = run_language_memory_slot_training_impact(
        output_path=output,
        config=MemorySlotTrainingImpactConfig(
            vocab_size=384,
            sampled_vocab_size=32,
            embedding_dim=12,
            state_dim=16,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=24,
            recurrent_gradient_horizon=4,
            memory_slot_count=4,
            bounded_memory_slot_candidate_count=2,
            active_memory_slot_count=1,
            sequence_length=8,
            batch_size=2,
            warmup_steps=1,
            repeats=2,
            gradient_clip_interval=1,
            device="cpu",
        ),
    )

    assert output.exists()
    assert report["surface"] == "marulho_language_memory_slot_training_impact.v1"
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["model_vocab_size"] == 384
    assert report["generation_vocab_size"] == report["tokenizer_vocab_size"]
    assert report["batch"]["tokens_per_optimizer_step"] == 16
    assert report["review"]["complete_training_step_impact"] is True
    assert report["review"]["includes_forward_backward_and_optimizer_step"] is True
    assert report["review"]["includes_memory_slot_gradient_evidence"] is True
    assert report["review"]["not_kernel_microbench_only"] is True
    assert report["review"]["sampled_loss_avoids_full_vocab_logits"] is True
    assert report["review"]["uses_sparse_vocab_optimizer"] is True
    assert report["review"]["uses_precomputed_sampled_vocab_hot_window"] is True
    assert report["review"]["bounded_memory_slots_trainable"] is True
    assert report["review"]["promotes_hot_path"] is False
    assert report["review"]["promotes_runtime_claim"] is False

    control = report["arms"]["memory_slots_disabled_control"]
    bounded = report["arms"]["bounded_memory_slots_enabled"]
    assert control["success"] is True
    assert bounded["success"] is True
    assert control["token_count"] == 32
    assert bounded["token_count"] == 32
    assert control["sampled_vocab_training"] is True
    assert bounded["sampled_vocab_training"] is True
    assert control["full_vocab_logits_materialized"] is False
    assert bounded["full_vocab_logits_materialized"] is False
    assert bounded["optimizer_policy"] == "AdamW_dense_core_plus_SparseAdam_vocab_rows"
    assert bounded["loss_evidence"]["loss_backend"] == (
        "torch_autograd_selected_lm_head_rows"
    )
    assert bounded["loss_evidence"]["precomputed_sampled_vocab_used"] is True
    assert bounded["loss_evidence"]["precomputed_target_positions_used"] is True
    assert bounded["loss_evidence"]["lm_head_weight_gradient_sparse"] is True
    assert bounded["loss_evidence"]["token_embedding_gradient_sparse"] is True
    assert bounded["memory_enabled"] is True
    assert bounded["total_slots"] == 4
    assert bounded["candidate_slot_count"] == 2
    assert bounded["active_slots_per_token"] == 1
    assert bounded["candidate_slots_scored"] == 32
    assert bounded["runs_all_slots"] is False
    assert bounded["memory_gate_readback"] is False
    assert bounded["memory_slot_nonzero_count"] > 0
    assert bounded["initial_memory_slot_gate_value"] == 0.0
    assert bounded["memory_slot_trainable_neutral_initialization"] is True
    assert bounded["memory_slot_gate_gradient"]["present"] is True
    assert bounded["memory_slot_gate_gradient"]["nonzero"] is True

    comparison = report["comparison"]
    assert comparison["control_success"] is True
    assert comparison["bounded_success"] is True
    assert comparison["bounded_memory_enabled"] is True
    assert comparison["bounded_avoids_all_slot_scan"] is True
    assert comparison["bounded_trainable_neutral_initialization"] is True
    assert comparison["bounded_memory_slot_gate_gradient_nonzero"] is True
    assert comparison["bounded_sampled_vocab_loss_without_full_logits"] is True
    assert comparison["evidence_status"] == "measured_bounded_memory_slot_training_impact"

    gate = report["promotion_gate"]
    assert gate["training_impact_available"] is True
    assert gate["bounded_memory_slots_enabled"] is True
    assert gate["bounded_avoids_all_slot_scan"] is True
    assert gate["sampled_vocab_training_without_full_logits"] is True
    assert gate["trainable_neutral_initialization"] is True
    assert gate["memory_gate_gradient_nonzero"] is True
    assert gate["complete_training_step_impact_available"] is True
    assert gate["promotes_hot_path"] is False
    assert gate["promotes_runtime_claim"] is False
