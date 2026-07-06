from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from marulho.core.language_sampled_vocab_ce_triton import (
    build_sampled_target_positions,
    build_sampled_vocab_ids,
)
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_checkpoint_evolution import (
    LanguageCheckpointEvolutionConfig,
    run_language_checkpoint_evolution,
)
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training import language_model as language_model_module
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    MarulhoSelectiveSpikingStateBlock,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    precompute_sampled_vocab_batches,
    save_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_column_split_proposal,
    build_language_structural_deep_sleep_proposal,
    build_language_structural_memory_slot_expansion_proposal,
    build_language_structural_merge_proposal,
    build_language_structural_prune_proposal,
    build_language_structural_plasticity_proposal,
    build_language_structural_retire_proposal,
    build_language_structural_route_bank_expansion_proposal,
    build_language_structural_synapse_bundle_proposal,
)


def _texts() -> list[str]:
    return [
        "marulho routes spike state into language evidence. " * 8,
        "runtime truth records checkpoints loss and replay boundaries. " * 8,
    ]


def test_byte_level_tokenizer_round_trips_and_restores_state() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    text = "MARULHO learns from runtime evidence.\n"

    token_ids = tokenizer.encode(text)
    restored = ByteLevelLanguageTokenizer.load_state_dict(tokenizer.state_dict())

    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(token_ids) == text
    assert restored.decode(token_ids) == text
    assert restored.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert tokenizer.state_dict()["external_dependency"] is False


def test_language_split_loader_reports_train_eval_hashes() -> None:
    tokenizer = ByteLevelLanguageTokenizer()

    split = build_language_model_splits(
        _texts(),
        tokenizer,
        sequence_length=16,
        eval_fraction=0.25,
    )

    assert split.report["surface"] == "marulho_language_train_eval_split.v1"
    assert split.report["owned_by_marulho"] is True
    assert split.report["external_dependency"] is False
    assert split.report["train_batch_count"] >= 1
    assert split.report["eval_batch_count"] >= 1
    assert split.report["tokenizer_hash"] == tokenizer.vocabulary_hash()
    assert split.report["train_split_hash"] != split.report["eval_split_hash"]
    assert split.train[0].input_ids.shape == split.train[0].target_ids.shape


def test_language_split_loader_packs_batched_windows() -> None:
    tokenizer = ByteLevelLanguageTokenizer()

    split = build_language_model_splits(
        _texts(),
        tokenizer,
        sequence_length=8,
        eval_fraction=0.25,
        stride=4,
        batch_size=3,
    )

    assert split.report["batch_size"] == 3
    assert split.report["window_count"] >= split.report["train_window_count"]
    assert split.train[0].input_ids.ndim == 2
    assert split.train[0].input_ids.shape[0] <= 3
    assert split.train[0].input_ids.shape[1] == 8
    assert split.train[0].target_ids.shape == split.train[0].input_ids.shape


def test_language_model_loss_and_spiking_telemetry_are_trainable() -> None:
    torch.manual_seed(7)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=12)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=24,
            spike_slope=4.0,
            adaptive_timestep_budget=2,
        )
    )

    result = model.next_token_loss(split.train[0].input_ids, split.train[0].target_ids)
    result["loss"].backward()

    assert result["loss"].detach().item() > 0
    assert result["logits"].shape[-1] == tokenizer.vocab_size
    assert result["telemetry"]["surface"] == "marulho_selective_spiking_state_block.v1"
    assert result["telemetry"]["active_language_path"] == "marulho_lm_head"
    assert result["telemetry"]["external_llm_used"] is False
    assert result["telemetry"]["normalization"] == "rmsnorm"
    assert result["telemetry"]["plif_state"] == "membrane_spikes_selective_state"
    assert result["telemetry"]["mixed_state_sequence_buffer_mode"] == (
        "stacked_mixed_state_list"
    )
    assert result["telemetry"]["adaptive_timestep_budget"] == 2
    assert result["telemetry"]["adaptive_step_count"] == split.train[0].input_ids.shape[1] * 2
    assert result["telemetry"]["input_dependent_leak"] is True
    assert result["telemetry"]["input_dependent_threshold"] is True
    assert result["telemetry"]["trainable_current_terms"] is True
    assert result["telemetry"]["recurrent_gradient_horizon"] == 0
    assert result["telemetry"]["truncated_bptt_applied"] is False
    assert result["telemetry"]["gradient_horizon_policy"] == "full_sequence_bptt"
    assert set(result["telemetry"]["state_cache_keys"]) == {
        "membrane",
        "spikes",
        "selective_state",
        "eligibility_trace",
    }
    assert 0.0 <= result["telemetry"]["spike_rate"] <= 1.0
    assert model.token_embedding.weight.grad is not None
    assert model.state_block.current_gain.grad is not None
    assert model.state_block.raw_leak.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert torch.isfinite(model.state_block.current_gain.grad).all()
    assert torch.isfinite(model.state_block.raw_leak.grad).all()


def test_language_model_recurrent_gradient_horizon_trains_with_bounded_bptt() -> None:
    torch.manual_seed(20260704)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        _texts(),
        tokenizer,
        sequence_length=12,
        batch_size=2,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=24,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=32,
            recurrent_gradient_horizon=4,
        )
    )

    result = model.next_token_loss(split.train[0].input_ids, split.train[0].target_ids)
    result["loss"].backward()

    telemetry = result["telemetry"]
    assert telemetry["recurrent_gradient_horizon"] == 4
    assert telemetry["truncated_bptt_applied"] is True
    assert telemetry["truncated_bptt_boundary_count"] == 2
    assert telemetry["gradient_horizon_policy"] == "bounded_recurrent_state_detach"
    assert result["loss"].detach().item() > 0.0
    assert model.token_embedding.weight.grad is not None
    assert model.state_block.current_gain.grad is not None
    assert model.state_block.raw_leak.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert torch.isfinite(model.state_block.current_gain.grad).all()
    assert torch.isfinite(model.state_block.raw_leak.grad).all()


def test_language_model_sampled_vocab_loss_skips_full_logits_and_trains_head() -> None:
    torch.manual_seed(20260704)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        _texts(),
        tokenizer,
        sequence_length=12,
        batch_size=2,
    )
    model_vocab_size = tokenizer.vocab_size + 128
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=16,
            state_dim=24,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=32,
            sampled_vocab_size=32,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
        )
    )

    batch = split.train[0]
    flat_targets = batch.target_ids.reshape(-1)
    sampled_vocab_ids = build_sampled_vocab_ids(
        flat_targets,
        vocab_size=model_vocab_size,
        sample_count=32,
        device=batch.target_ids.device,
        validate_ids=False,
    )
    sampled_target_positions = build_sampled_target_positions(
        flat_targets,
        sampled_vocab_ids,
        device=batch.target_ids.device,
        validate_targets=True,
    )
    default_result = model.next_token_loss(batch.input_ids, batch.target_ids)
    result = model.next_token_loss(
        batch.input_ids,
        batch.target_ids,
        sampled_vocab_ids=sampled_vocab_ids,
        sampled_target_positions=sampled_target_positions,
    )
    lean_result = model.next_token_loss(
        batch.input_ids,
        batch.target_ids,
        sampled_vocab_ids=sampled_vocab_ids,
        sampled_target_positions=sampled_target_positions,
        return_evidence=False,
    )
    result["loss"].backward()

    assert result["loss"].detach().item() > 0
    torch.testing.assert_close(
        result["loss"].detach(),
        default_result["loss"].detach(),
    )
    torch.testing.assert_close(
        lean_result["loss"].detach(),
        result["loss"].detach(),
    )
    assert lean_result["logits"] is None
    assert lean_result["loss_kind"] == "sampled_adaptive_vocab_cross_entropy"
    assert "loss_evidence" not in lean_result
    assert "telemetry" not in lean_result
    assert result["logits"] is None
    assert result["loss_kind"] == "sampled_adaptive_vocab_cross_entropy"
    evidence = result["loss_evidence"]
    assert evidence["full_vocab_logits_materialized"] is False
    assert evidence["sampled_vocab_training"] is True
    assert evidence["loss_backend"] == "torch_autograd_selected_lm_head_rows"
    assert evidence["lm_head_weight_gradient_sparse"] is True
    assert evidence["token_embedding_gradient_sparse"] is True
    assert evidence["per_batch_target_membership_cpu_sync"] is False
    assert evidence["sampled_vocab_id_source"] == "precomputed_batch_sampled_vocab_ids"
    assert evidence["sampled_target_position_source"] == (
        "precomputed_batch_target_positions"
    )
    assert evidence["precomputed_sampled_vocab_used"] is True
    assert evidence["precomputed_target_positions_used"] is True
    assert evidence["actual_sampled_vocab_size"] >= 32
    assert evidence["actual_sampled_vocab_size"] < model_vocab_size
    assert evidence["model_vocab_size"] == model_vocab_size
    assert result["telemetry"]["vocab_size"] == model_vocab_size
    assert result["telemetry"]["vocab_loss"] == evidence
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.is_sparse
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad.is_sparse
    updated_rows = int(model.lm_head.weight.grad.coalesce().indices()[0].unique().numel())
    assert 0 < updated_rows <= int(evidence["actual_sampled_vocab_size"])
    assert torch.isfinite(model.token_embedding.weight.grad.coalesce().values()).all()
    assert torch.isfinite(model.lm_head.weight.grad.coalesce().values()).all()


def test_language_model_pair_fuses_sampled_vocab_loss_with_shared_rows() -> None:
    torch.manual_seed(20260704)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        _texts(),
        tokenizer,
        sequence_length=12,
        batch_size=2,
    )
    model_vocab_size = tokenizer.vocab_size + 128
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=16,
            state_dim=24,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=32,
            sampled_vocab_size=32,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
        )
    )
    update_source = split.train[0]
    replay_source = split.train[1]
    update_targets = update_source.target_ids.reshape(-1)
    replay_targets = replay_source.target_ids.reshape(-1)
    combined_targets = torch.cat((update_targets, replay_targets), dim=0)
    paired_sampled_vocab_ids = build_sampled_vocab_ids(
        combined_targets,
        vocab_size=model_vocab_size,
        sample_count=32,
        device=combined_targets.device,
        validate_ids=False,
    )
    paired_positions = build_sampled_target_positions(
        combined_targets,
        paired_sampled_vocab_ids,
        device=combined_targets.device,
        validate_targets=True,
    )
    update_positions = paired_positions[: int(update_targets.numel())]
    replay_positions = paired_positions[int(update_targets.numel()) :]
    update_batch = LanguageBatch(
        input_ids=update_source.input_ids,
        target_ids=update_source.target_ids,
        sampled_vocab_ids=paired_sampled_vocab_ids,
        sampled_target_positions=update_positions,
    )
    replay_batch = LanguageBatch(
        input_ids=replay_source.input_ids,
        target_ids=replay_source.target_ids,
        sampled_vocab_ids=paired_sampled_vocab_ids,
        sampled_target_positions=replay_positions,
    )

    separate_result = model.next_token_loss_pair(
        update_batch,
        replay_batch,
        replay_loss_weight=0.35,
    )
    fused_result = model.next_token_loss_pair(
        update_batch,
        replay_batch,
        replay_loss_weight=0.35,
        paired_sampled_vocab_ids=paired_sampled_vocab_ids,
        paired_sampled_target_ids=combined_targets,
        paired_sampled_target_positions=paired_positions,
    )

    assert separate_result["paired_sampled_vocab_loss_fused"] is False
    assert fused_result["paired_sampled_vocab_loss_fused"] is True
    assert fused_result["loss_kind"] == "paired_sampled_adaptive_vocab_cross_entropy"
    torch.testing.assert_close(
        fused_result["update_loss"].detach(),
        separate_result["update_loss"].detach(),
    )
    torch.testing.assert_close(
        fused_result["replay_loss"].detach(),
        separate_result["replay_loss"].detach(),
    )
    torch.testing.assert_close(
        fused_result["loss"].detach(),
        separate_result["loss"].detach(),
    )


def test_padded_vocab_generation_limits_decode_rows_and_restores_checkpoint(tmp_path) -> None:
    torch.manual_seed(20260704)
    tokenizer = ByteLevelLanguageTokenizer()
    model_vocab_size = tokenizer.vocab_size + 128
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=12,
            state_dim=16,
            generation_vocab_size=tokenizer.vocab_size,
        )
    )
    with torch.no_grad():
        model.lm_head.bias[tokenizer.vocab_size :].fill_(1_000_000.0)

    prompt = torch.tensor(tokenizer.encode("marulho", add_eos=False), dtype=torch.long)
    generation = model.generate(prompt, max_new_tokens=4, eos_id=None)
    generated_ids = generation["generated_ids"].reshape(-1)
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "padded-language-model.pt",
        model,
        tokenizer,
        metadata={"policy": "padded-vocab-decode-limit"},
    )
    restored, restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert generated_ids.max().item() < tokenizer.vocab_size
    assert generation["generation_decode"]["model_vocab_size"] == model_vocab_size
    assert generation["generation_decode"]["generation_vocab_size"] == tokenizer.vocab_size
    assert generation["generation_decode"]["full_model_vocab_logits_materialized"] is False
    assert restored.config.vocab_size == model_vocab_size
    assert restored.config.generation_vocab_size == tokenizer.vocab_size
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata["policy"] == "padded-vocab-decode-limit"


def test_generation_no_repeat_ngram_reports_decode_controls() -> None:
    torch.manual_seed(20260704)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=16,
            embedding_dim=8,
            state_dim=12,
        )
    )
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[5] = 10.0
    prompt = torch.tensor([1], dtype=torch.long)

    greedy = model.generate(prompt, max_new_tokens=3, eos_id=None)
    controlled = model.generate(
        prompt,
        max_new_tokens=3,
        eos_id=None,
        repetition_penalty=1.2,
        no_repeat_ngram_size=1,
    )
    greedy_tail = greedy["generated_ids"].reshape(-1).tolist()[1:]
    controlled_tail = controlled["generated_ids"].reshape(-1).tolist()[1:]
    decode = controlled["generation_decode"]

    assert greedy_tail == [5, 5, 5]
    assert len(set(controlled_tail)) == len(controlled_tail)
    assert decode["repetition_penalty_applied"] is True
    assert decode["repetition_penalty"] == 1.2
    assert decode["no_repeat_ngram_applied"] is True
    assert decode["no_repeat_ngram_size"] == 1
    assert decode["decode_controls_backend"] == "torch_device_tensor"
    assert decode["decode_controls_cpu_token_copy"] is False
    assert decode["repetition_penalty_adjusted_token_count"] > 0
    assert decode["no_repeat_ngram_banned_token_count"] > 0
    assert decode["decode_control_fallback_count"] == 0
    assert controlled["external_llm_used"] is False


def test_padded_vocab_checkpoint_requires_decode_policy(tmp_path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size + 8,
            embedding_dim=8,
            state_dim=12,
        )
    )

    with pytest.raises(ValueError, match="generation_vocab_size"):
        save_language_model_checkpoint(tmp_path / "invalid-padded.pt", model, tokenizer)


def test_selective_state_cache_matches_full_sequence_suffix() -> None:
    torch.manual_seed(9)
    tokenizer = ByteLevelLanguageTokenizer()
    token_ids = torch.tensor(
        [tokenizer.encode("streaming state cache stays causal and exact.")],
        dtype=torch.long,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            adaptive_timestep_budget=2,
        )
    )
    model.eval()
    split_at = token_ids.shape[1] // 2

    with torch.no_grad():
        full = model(token_ids)
        prefix = model(token_ids[:, :split_at])
        suffix = model(token_ids[:, split_at:], state=prefix["state"])

    assert set(prefix["state"]) == {
        "membrane",
        "spikes",
        "selective_state",
        "eligibility_trace",
    }
    assert suffix["telemetry"]["adaptive_timestep_budget"] == 2
    torch.testing.assert_close(
        full["logits"][:, split_at:, :],
        suffix["logits"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_selective_state_block_vectorized_forward_matches_step_loop() -> None:
    torch.manual_seed(11)
    block = MarulhoSelectiveSpikingStateBlock(
        input_dim=12,
        state_dim=20,
        adaptive_timestep_budget=2,
    )
    inputs = torch.randn(3, 7, 12)

    with torch.no_grad():
        full_hidden, full_state, full_telemetry = block(inputs)
        state = None
        step_outputs = []
        for offset in range(inputs.shape[1]):
            hidden, state, _telemetry = block.step(
                inputs[:, offset, :],
                state,
                collect_telemetry=False,
            )
            step_outputs.append(hidden)
        stepped_hidden = torch.stack(step_outputs, dim=1)

    torch.testing.assert_close(full_hidden, stepped_hidden, rtol=1e-6, atol=1e-6)
    assert state is not None
    for key, value in full_state.items():
        torch.testing.assert_close(value, state[key], rtol=1e-6, atol=1e-6)
    assert full_telemetry["state_block_projection_mode"] == (
        "batched_token_and_state_output_projection_recurrent_loop"
    )
    assert full_telemetry["mixed_state_sequence_buffer_mode"] == (
        "stacked_mixed_state_list"
    )


def test_selective_state_block_deferred_eligibility_matches_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260705)
    block = MarulhoSelectiveSpikingStateBlock(
        input_dim=12,
        state_dim=20,
        adaptive_timestep_budget=1,
    )
    inputs = torch.randn(3, 7, 12)

    monkeypatch.setenv("MARULHO_LANGUAGE_STATE_BLOCK_DEFER_ELIGIBILITY_NO_GRAD", "0")
    with torch.no_grad():
        inline_hidden, inline_state, inline_telemetry = block(inputs)
    monkeypatch.setenv("MARULHO_LANGUAGE_STATE_BLOCK_DEFER_ELIGIBILITY_NO_GRAD", "1")
    with torch.no_grad():
        deferred_hidden, deferred_state, deferred_telemetry = block(inputs)

    torch.testing.assert_close(deferred_hidden, inline_hidden, rtol=1e-6, atol=1e-6)
    for key, value in inline_state.items():
        torch.testing.assert_close(deferred_state[key], value, rtol=1e-6, atol=1e-6)
    assert inline_telemetry["eligibility_trace_update_mode"] == "inline_plif_update"
    assert deferred_telemetry["eligibility_trace_update_mode"] == (
        "deferred_sequence_scan_no_grad"
    )
    assert deferred_telemetry["eligibility_trace_sequence_buffer_mode"] == (
        "spike_sequence_buffer"
    )


def test_language_model_forward_step_matches_single_token_forward() -> None:
    torch.manual_seed(10)
    tokenizer = ByteLevelLanguageTokenizer()
    token_ids = torch.tensor(
        [tokenizer.encode("single token streaming stays equivalent.")],
        dtype=torch.long,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=3,
            expert_hidden_dim=24,
        )
    )
    model.eval()
    split_at = max(1, token_ids.shape[1] // 2)

    with torch.no_grad():
        prefix = model(token_ids[:, :split_at])
        suffix = token_ids[:, split_at : split_at + 1]
        full_step = model(suffix, state=prefix["state"])
        stream_step = model.forward_step(suffix, state=prefix["state"])

    torch.testing.assert_close(stream_step["logits"], full_step["logits"])
    for key, value in full_step["state"].items():
        torch.testing.assert_close(stream_step["state"][key], value)
    assert stream_step["telemetry"]["active_language_path"] == "marulho_lm_head"
    assert stream_step["telemetry"]["external_llm_used"] is False


def test_language_model_routes_bounded_sparse_experts_without_all_column_scan() -> None:
    torch.manual_seed(13)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=6,
            active_expert_count=2,
            route_candidate_count=3,
            expert_hidden_dim=24,
        )
    )

    result = model.next_token_loss(split.train[0].input_ids, split.train[0].target_ids)
    result["loss"].backward()
    routing = result["telemetry"]["routing"]

    assert routing["surface"] == "marulho_routed_language_experts.v1"
    assert routing["enabled"] is True
    assert routing["route_plan_source"] == "token_hash_candidate_bank"
    assert routing["candidate_id_source"] == "awake_index_select"
    assert routing["all_awake_candidate_fastpath"] is False
    assert routing["total_columns"] == 6
    assert 1 <= routing["active_columns"] <= 6
    assert routing["active_expert_count_per_token"] == 2
    assert routing["route_candidate_count"] == 3
    assert routing["candidate_rows_scored"] == split.train[0].input_ids.numel() * 3
    assert routing["output_candidate_count"] == 2
    assert routing["runs_all_columns"] is False
    assert routing["fallback_reason"] is None
    assert routing["route_selection_backend"] == "torch_grad_route_topk"
    assert routing["expert_dispatch_backend"] == (
        "torch_selected_expert_batched_matmul_dispatch"
    )
    assert routing["route_device"] == "cpu"
    assert routing["route_latency_ms"] >= 0.0
    assert routing["active_parameters_per_token"] > 0
    assert model.routed_experts.route_keys.grad is not None
    assert torch.isfinite(model.routed_experts.route_keys.grad).all()

    all_awake = model.next_token_loss(
        split.train[0].input_ids,
        split.train[0].target_ids,
        assume_no_sleeping_experts=True,
    )
    all_awake_routing = all_awake["telemetry"]["routing"]
    assert all_awake_routing["route_plan_source"] == (
        "token_hash_candidate_bank_all_awake_direct_modulo"
    )
    assert all_awake_routing["candidate_id_source"] == "all_awake_direct_expert_ids"
    assert all_awake_routing["all_awake_candidate_fastpath"] is True
    assert all_awake_routing["candidate_rows_scored"] == (
        split.train[0].input_ids.numel() * 3
    )
    cached_batches, precompute_report = precompute_sampled_vocab_batches(
        model,
        (split.train[0],),
        assume_no_sleeping_experts=True,
    )
    precomputed = model.next_token_loss(
        cached_batches[0].input_ids,
        cached_batches[0].target_ids,
        assume_no_sleeping_experts=True,
        route_candidate_ids=cached_batches[0].route_candidate_ids,
    )
    precomputed_routing = precomputed["telemetry"]["routing"]
    torch.testing.assert_close(precomputed["loss"], all_awake["loss"])
    assert precompute_report["route_candidate_precompute"]["enabled"] is True
    assert precompute_report["route_candidate_precompute"]["candidate_id_source"] == (
        "precomputed_batch_route_candidate_ids"
    )
    assert precomputed_routing["precomputed_candidate_ids_used"] is True
    assert precomputed_routing["candidate_id_source"] == "all_awake_direct_expert_ids"
    assert precomputed_routing["all_awake_candidate_fastpath"] is True


def test_language_model_reads_bounded_memory_slots_without_all_slot_scan(tmp_path) -> None:
    torch.manual_seed(14)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    base_config = LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        state_dim=20,
        expert_count=2,
        active_expert_count=1,
        route_candidate_count=2,
    )
    memory_config = LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        state_dim=20,
        expert_count=2,
        active_expert_count=1,
        route_candidate_count=2,
        memory_slot_count=4,
        memory_slot_candidate_count=2,
        active_memory_slot_count=1,
    )
    torch.manual_seed(14)
    disabled_model = MarulhoLanguageModel(base_config)
    torch.manual_seed(14)
    model = MarulhoLanguageModel(memory_config)
    disabled_logits = disabled_model.forward(
        split.train[0].input_ids,
        collect_telemetry=False,
        decode_vocab_only=True,
    )["logits"]
    memory_logits = model.forward(
        split.train[0].input_ids,
        collect_telemetry=False,
        decode_vocab_only=True,
    )["logits"]

    result = model.next_token_loss(split.train[0].input_ids, split.train[0].target_ids)
    result["loss"].backward()
    memory = result["telemetry"]["memory"]
    cached_batches, precompute_report = precompute_sampled_vocab_batches(
        model,
        (split.train[0],),
    )
    precomputed_result = model.next_token_loss(
        cached_batches[0].input_ids,
        cached_batches[0].target_ids,
        memory_candidate_ids=cached_batches[0].memory_candidate_ids,
    )
    precomputed_memory = precomputed_result["telemetry"]["memory"]
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-memory-slots.pt",
        model,
        tokenizer,
        metadata={"transaction": "memory_slots_enabled"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert memory["surface"] == "marulho_language_memory_slots.v1"
    assert memory["enabled"] is True
    assert memory["total_slots"] == 4
    assert memory["candidate_slot_count"] == 2
    assert memory["active_slots_per_token"] == 1
    assert memory["candidate_slots_scored"] == split.train[0].input_ids.numel() * 2
    assert memory["runs_all_slots"] is False
    assert memory["fallback_reason"] is None
    assert memory["candidate_id_source"] == "token_hash_memory_slot_bank"
    assert memory["memory_gate_readback"] is False
    assert memory["memory_device"] == "cpu"
    assert memory["memory_slot_initialization"] == "nonzero_slots_zero_gate"
    assert memory["memory_slot_init_std"] == 0.02
    assert precompute_report["memory_candidate_precompute"]["enabled"] is True
    assert precompute_report["memory_candidate_precompute"]["candidate_id_source"] == (
        "precomputed_batch_memory_candidate_ids"
    )
    assert cached_batches[0].memory_candidate_ids is not None
    assert precomputed_memory["candidate_id_source"] == (
        "precomputed_batch_memory_candidate_ids"
    )
    assert precomputed_memory["precomputed_candidate_ids_used"] is True
    assert precomputed_memory["candidate_slots_scored"] == memory[
        "candidate_slots_scored"
    ]
    torch.testing.assert_close(precomputed_result["loss"], result["loss"])
    torch.testing.assert_close(memory_logits, disabled_logits)
    assert model.memory_slots is not None
    assert torch.count_nonzero(model.memory_slots).item() > 0
    assert model.memory_slot_gate is not None
    assert model.memory_slot_gate.detach().item() == 0.0
    assert model.memory_slot_gate.grad is not None
    assert model.memory_slot_gate.grad.detach().abs().item() > 0.0
    assert model.memory_slots.grad is not None
    assert metadata["transaction"] == "memory_slots_enabled"
    assert restored_model.config.memory_slot_count == 4
    assert restored_model.config.memory_slot_candidate_count == 2
    assert restored_model.config.active_memory_slot_count == 1
    assert restored_model.memory_slots is not None
    assert tuple(restored_model.memory_slots.shape) == (4, 20)


def test_language_model_forward_step_uses_bounded_memory_slots() -> None:
    torch.manual_seed(141)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    base_config = LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        state_dim=20,
        expert_count=2,
        active_expert_count=1,
        route_candidate_count=2,
    )
    memory_config = LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        state_dim=20,
        expert_count=2,
        active_expert_count=1,
        route_candidate_count=2,
        memory_slot_count=4,
        memory_slot_candidate_count=2,
        active_memory_slot_count=1,
    )
    torch.manual_seed(141)
    disabled_model = MarulhoLanguageModel(base_config)
    torch.manual_seed(141)
    model = MarulhoLanguageModel(memory_config)
    token_ids = split.train[0].input_ids[:, 0]

    disabled_result = disabled_model.forward_step(
        token_ids,
        collect_telemetry=False,
        decode_vocab_only=True,
    )
    memory_result = model.forward_step(
        token_ids,
        collect_telemetry=True,
        decode_vocab_only=True,
    )
    memory = memory_result["telemetry"]["memory"]

    torch.testing.assert_close(memory_result["logits"], disabled_result["logits"])
    assert memory["surface"] == "marulho_language_memory_slots.v1"
    assert memory["enabled"] is True
    assert memory["total_slots"] == 4
    assert memory["candidate_slot_count"] == 2
    assert memory["active_slots_per_token"] == 1
    assert memory["candidate_slots_scored"] == token_ids.numel() * 2
    assert memory["runs_all_slots"] is False
    assert memory["candidate_id_source"] == "token_hash_memory_slot_bank"
    assert memory["memory_gate_readback"] is False
    assert memory_result["telemetry"]["generation_decode"][
        "full_model_vocab_logits_materialized"
    ] is True


def test_language_model_sleeping_experts_are_skipped_by_route_candidates() -> None:
    torch.manual_seed(15)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=4,
            expert_hidden_dim=24,
        )
    )
    with torch.no_grad():
        model.routed_experts.sleeping_expert_mask[3] = True

    result = model.next_token_loss(split.train[0].input_ids, split.train[0].target_ids)
    routing = result["telemetry"]["routing"]

    assert routing["surface"] == "marulho_routed_language_experts.v1"
    assert routing["sleep_filter_applied"] is True
    assert routing["sleeping_columns"] == 1
    assert routing["awake_columns"] == 3
    assert routing["sleeping_expert_ids"] == [3]
    assert routing["candidate_id_source"] == "awake_index_select"
    assert routing["all_awake_candidate_fastpath"] is False
    assert routing["route_candidate_count"] == 3
    assert routing["candidate_rows_scored"] == split.train[0].input_ids.numel() * 3
    assert routing["runs_all_columns"] is False
    assert routing["fallback_reason"] is None


def test_language_structural_plasticity_expands_experts_with_checkpoint(tmp_path) -> None:
    torch.manual_seed(23)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    routing_evidence = {
        "surface": "marulho_routed_language_experts.v1",
        "total_columns": 2,
        "active_columns": 2,
        "candidate_rows_scored": 20,
        "runs_all_columns": False,
    }

    proposal = build_language_structural_plasticity_proposal(
        model,
        routing_evidence=routing_evidence,
        config=LanguageStructuralPlasticityConfig(
            route_saturation_threshold=0.5,
            max_added_experts=2,
        ),
    )
    grown_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-structure-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(max_eval_loss_delta=10.0),
    )

    assert proposal["surface"] == "marulho_language_structural_plasticity_proposal.v1"
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["requires_operator_approval"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutates_runtime_state"] is True
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert Path(report["checkpoint"]["path"]).exists()
    assert report["mutation"]["source_expert_count"] == 2
    assert report["mutation"]["target_expert_count"] > 2
    assert grown_model.config.expert_count == report["mutation"]["target_expert_count"]
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is True


def test_language_structural_plasticity_splits_column_with_checkpoint(tmp_path) -> None:
    torch.manual_seed(231)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    parent_parameters = [
        parameter.detach().clone()
        for parameter in model.routed_experts.experts[1].parameters()
    ]

    proposal = build_language_structural_column_split_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "split_candidate_expert_ids": [1],
            "expert_loads": [0.1, 0.95, 0.2, 0.3],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_added_experts=2,
            max_split_experts=1,
            split_load_threshold=0.8,
        ),
    )
    split_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-column-split-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_added_experts=2,
            max_split_experts=1,
            max_eval_loss_delta=100.0,
        ),
    )
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-column-split.pt",
        split_model,
        tokenizer,
        metadata={"transaction": "column_split"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["proposal"]["proposal_kind"] == "column_split"
    assert proposal["proposal"]["split_expert_ids"] == [1]
    assert proposal["proposal"]["child_expert_ids"] == [4]
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "column_split"
    assert report["mutation"]["source_expert_count"] == 4
    assert report["mutation"]["target_expert_count"] == 5
    assert report["mutation"]["split_expert_ids"] == [1]
    assert report["mutation"]["split_child_expert_ids"] == [4]
    assert report["mutation"]["parent_child_expert_pairs"] == [[1, 4]]
    assert report["mutation"]["added_expert_count"] == 1
    assert split_model.config.expert_count == 5
    assert split_model.routed_experts.sleeping_expert_ids() == []
    for expected, actual in zip(
        parent_parameters,
        split_model.routed_experts.experts[4].parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.detach().cpu(), expected)
    assert metadata["transaction"] == "column_split"
    assert restored_model.config.expert_count == 5
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_column_split_promotion"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_prunes_experts_with_checkpoint(tmp_path) -> None:
    torch.manual_seed(24)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )

    proposal = build_language_structural_prune_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "active_expert_ids": [0],
            "inactive_expert_ids": [2],
            "expert_utilities": [0.8, 0.2, 0.0],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_pruned_experts=1,
            prune_utility_threshold=0.05,
        ),
    )
    pruned_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-prune-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_pruned_experts=1,
            max_eval_loss_delta=10.0,
        ),
    )

    assert proposal["surface"] == "marulho_language_structural_plasticity_proposal.v1"
    assert proposal["proposal"]["proposal_kind"] == "expert_prune"
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["min_expert_count_preserved"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "expert_prune"
    assert report["mutation"]["source_expert_count"] == 3
    assert report["mutation"]["target_expert_count"] == 2
    assert report["mutation"]["pruned_expert_count"] == 1
    assert report["mutation"]["pruned_expert_ids"] == [2]
    assert pruned_model.config.expert_count == 2
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_prune_promotion"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_retires_experts_with_checkpoint(tmp_path) -> None:
    torch.manual_seed(241)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )

    proposal = build_language_structural_retire_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "retire_candidate_expert_ids": [3],
            "dead_spike_expert_ids": [3],
            "expert_utilities": [0.8, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_retired_experts=1,
            prune_utility_threshold=0.05,
        ),
    )
    retired_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-retire-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_retired_experts=1,
            max_eval_loss_delta=100.0,
        ),
    )
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-retire.pt",
        retired_model,
        tokenizer,
        metadata={"transaction": "expert_retire"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["proposal"]["proposal_kind"] == "expert_retire"
    assert proposal["proposal"]["retired_expert_ids"] == [3]
    assert proposal["proposal"]["retained_expert_ids"] == [0, 1, 2]
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["terminal_retirement_reviewable"] is True
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "expert_retire"
    assert report["mutation"]["source_expert_count"] == 4
    assert report["mutation"]["target_expert_count"] == 3
    assert report["mutation"]["retired_expert_count"] == 1
    assert report["mutation"]["retired_expert_ids"] == [3]
    assert report["mutation"]["pruned_expert_count"] == 0
    assert retired_model.config.expert_count == 3
    assert metadata["transaction"] == "expert_retire"
    assert restored_model.config.expert_count == 3
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_retire_promotion"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_prune_promotion"] is False


def test_language_structural_plasticity_merges_experts_with_checkpoint(tmp_path) -> None:
    torch.manual_seed(25)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    with torch.no_grad():
        model.routed_experts.route_keys[1].fill_(1.0)
        model.routed_experts.route_keys[2].fill_(3.0)

    proposal = build_language_structural_merge_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "duplicate_expert_pairs": [[1, 2]],
            "expert_pair_similarities": {"1,2": 0.99},
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_merged_expert_pairs=1,
            merge_similarity_threshold=0.95,
        ),
    )
    merged_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-merge-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_merged_expert_pairs=1,
            max_eval_loss_delta=10.0,
        ),
    )

    assert proposal["surface"] == "marulho_language_structural_plasticity_proposal.v1"
    assert proposal["proposal"]["proposal_kind"] == "expert_merge"
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["active_expert_count_preserved"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "expert_merge"
    assert report["mutation"]["source_expert_count"] == 4
    assert report["mutation"]["target_expert_count"] == 3
    assert report["mutation"]["merged_expert_group_count"] == 1
    assert report["mutation"]["structural_reduction_count"] == 1
    assert report["mutation"]["merged_expert_groups"] == [[1, 2]]
    assert report["mutation"]["removed_expert_ids"] == [2]
    assert report["mutation"]["pruned_expert_count"] == 0
    assert merged_model.config.expert_count == 3
    torch.testing.assert_close(
        merged_model.routed_experts.route_keys[1],
        torch.full_like(merged_model.routed_experts.route_keys[1], 2.0),
    )
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_merge_promotion"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_prune_promotion"] is False
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_deep_sleeps_experts_with_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(26)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=4,
        )
    )

    proposal = build_language_structural_deep_sleep_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "stale_expert_ids": [3],
            "low_activation_expert_ids": [3],
            "expert_utilities": [0.7, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_deep_sleep_experts=1,
            deep_sleep_utility_threshold=0.10,
        ),
    )
    slept_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-deep-sleep-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_deep_sleep_experts=1,
            max_eval_loss_delta=10.0,
        ),
    )
    sleep_eval = evaluate_language_model(slept_model, split.eval)
    sleep_routing = sleep_eval["spike_telemetry"]["routing"]
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-deep-sleep.pt",
        slept_model,
        tokenizer,
        metadata={"transaction": "deep_sleep"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["surface"] == "marulho_language_structural_plasticity_proposal.v1"
    assert proposal["proposal"]["proposal_kind"] == "expert_deep_sleep"
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["deep_sleep_reduces_awake_candidates"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "expert_deep_sleep"
    assert report["mutation"]["source_expert_count"] == 4
    assert report["mutation"]["target_expert_count"] == 4
    assert report["mutation"]["deep_sleep_expert_count"] == 1
    assert report["mutation"]["deep_sleep_expert_ids"] == [3]
    assert report["mutation"]["sleeping_expert_ids_after"] == [3]
    assert report["mutation"]["awake_expert_count_after"] == 3
    assert slept_model.config.expert_count == 4
    assert slept_model.routed_experts.sleeping_expert_ids() == [3]
    assert sleep_routing["sleeping_expert_ids"] == [3]
    assert sleep_routing["awake_columns"] == 3
    assert sleep_routing["candidate_rows_scored"] == split.eval[0].input_ids.numel() * 3
    assert sleep_routing["runs_all_columns"] is False
    assert metadata["transaction"] == "deep_sleep"
    assert restored_model.routed_experts.sleeping_expert_ids() == [3]
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_deep_sleep_promotion"] is True
    assert report["promotion_gate"]["eligible_for_reviewed_prune_promotion"] is False
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_expands_route_bank_with_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(27)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=5,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )

    proposal = build_language_structural_route_bank_expansion_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 5,
            "active_columns": 2,
            "route_candidate_count": 2,
            "output_candidate_count": 1,
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
            "route_bank_pressure": True,
        },
        config=LanguageStructuralPlasticityConfig(
            route_saturation_threshold=0.5,
            max_route_candidate_growth=2,
        ),
    )
    route_bank_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-route-bank-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_route_candidate_growth=2,
            max_eval_loss_delta=100.0,
        ),
    )
    route_bank_eval = evaluate_language_model(route_bank_model, split.eval)
    route_bank_routing = route_bank_eval["spike_telemetry"]["routing"]
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-route-bank.pt",
        route_bank_model,
        tokenizer,
        metadata={"transaction": "route_bank_expansion"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["surface"] == "marulho_language_structural_plasticity_proposal.v1"
    assert proposal["proposal"]["proposal_kind"] == "route_bank_expansion"
    assert proposal["proposal"]["source_route_candidate_count"] == 2
    assert proposal["proposal"]["target_route_candidate_count"] == 4
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["avoids_all_column_route_scan"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "route_bank_expansion"
    assert report["mutation"]["source_expert_count"] == 5
    assert report["mutation"]["target_expert_count"] == 5
    assert report["mutation"]["source_route_candidate_count"] == 2
    assert report["mutation"]["target_route_candidate_count"] == 4
    assert report["mutation"]["route_bank_candidate_count_delta"] == 2
    assert route_bank_model.config.expert_count == 5
    assert route_bank_model.config.route_candidate_count == 4
    assert route_bank_routing["route_candidate_count"] == 4
    assert route_bank_routing["candidate_rows_scored"] == (
        split.eval[0].input_ids.numel() * 4
    )
    assert route_bank_routing["runs_all_columns"] is False
    assert metadata["transaction"] == "route_bank_expansion"
    assert restored_model.config.route_candidate_count == 4
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"][
        "eligible_for_reviewed_route_bank_expansion_promotion"
    ] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_grows_synapse_bundle_with_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(28)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
        )
    )
    source_first_weights = [
        expert[0].weight.detach().clone()
        for expert in model.routed_experts.experts
    ]
    source_first_biases = [
        expert[0].bias.detach().clone()
        for expert in model.routed_experts.experts
    ]
    source_second_weights = [
        expert[2].weight.detach().clone()
        for expert in model.routed_experts.experts
    ]
    source_second_biases = [
        expert[2].bias.detach().clone()
        for expert in model.routed_experts.experts
    ]

    proposal = build_language_structural_synapse_bundle_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "synapse_bundle_pressure": True,
            "high_surprise_expert_ids": [1],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_synapse_bundle_hidden_growth=8,
        ),
    )
    grown_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-synapse-bundle-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_synapse_bundle_hidden_growth=8,
            max_eval_loss_delta=100.0,
        ),
    )
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-synapse-bundle.pt",
        grown_model,
        tokenizer,
        metadata={"transaction": "synapse_bundle_growth"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["proposal"]["proposal_kind"] == "synapse_bundle_growth"
    assert proposal["proposal"]["source_expert_hidden_dim"] == 24
    assert proposal["proposal"]["target_expert_hidden_dim"] == 32
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["new_bundle_initially_neutral"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "synapse_bundle_growth"
    assert report["mutation"]["source_expert_hidden_dim"] == 24
    assert report["mutation"]["target_expert_hidden_dim"] == 32
    assert report["mutation"]["synapse_bundle_hidden_growth"] == 8
    assert grown_model.config.expert_count == 3
    assert grown_model.config.expert_hidden_dim == 32
    for expert_id, expert in enumerate(grown_model.routed_experts.experts):
        torch.testing.assert_close(expert[0].weight[:24], source_first_weights[expert_id])
        torch.testing.assert_close(expert[0].bias[:24], source_first_biases[expert_id])
        torch.testing.assert_close(expert[2].weight[:, :24], source_second_weights[expert_id])
        torch.testing.assert_close(expert[2].bias, source_second_biases[expert_id])
        assert torch.count_nonzero(expert[0].weight[24:]).item() == 0
        assert torch.count_nonzero(expert[0].bias[24:]).item() == 0
        assert torch.count_nonzero(expert[2].weight[:, 24:]).item() == 0
    assert metadata["transaction"] == "synapse_bundle_growth"
    assert restored_model.config.expert_hidden_dim == 32
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"][
        "eligible_for_reviewed_synapse_bundle_promotion"
    ] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_language_structural_plasticity_expands_memory_slots_with_checkpoint(
    tmp_path,
) -> None:
    torch.manual_seed(30)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=10)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            memory_slot_count=0,
            memory_slot_candidate_count=0,
            active_memory_slot_count=1,
        )
    )
    baseline_logits = model.forward(
        split.eval[0].input_ids,
        collect_telemetry=False,
        decode_vocab_only=True,
    )["logits"]

    proposal = build_language_structural_memory_slot_expansion_proposal(
        model,
        routing_evidence={
            "surface": "marulho_language_memory_slots.v1",
            "memory_slot_pressure": True,
            "novel_concept_cluster": True,
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_memory_slot_growth=4,
            max_memory_slot_candidate_count=2,
        ),
    )
    memory_model, report = apply_language_structural_plasticity_transaction(
        model,
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "lm-memory-slot-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_memory_slot_growth=4,
            max_memory_slot_candidate_count=2,
            max_eval_loss_delta=100.0,
        ),
    )
    expanded_logits = memory_model.forward(
        split.eval[0].input_ids,
        collect_telemetry=False,
        decode_vocab_only=True,
    )["logits"]
    memory_train_result = memory_model.next_token_loss(
        split.train[0].input_ids,
        split.train[0].target_ids,
    )
    memory_train_result["loss"].backward()
    memory_eval = evaluate_language_model(memory_model, split.eval)
    memory = memory_eval["spike_telemetry"]["memory"]
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "lm-memory-slot-expanded.pt",
        memory_model,
        tokenizer,
        metadata={"transaction": "memory_slot_expansion"},
    )
    restored_model, _restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )

    assert proposal["proposal"]["proposal_kind"] == "memory_slot_expansion"
    assert proposal["proposal"]["source_memory_slot_count"] == 0
    assert proposal["proposal"]["target_memory_slot_count"] == 4
    assert proposal["proposal"]["target_memory_slot_candidate_count"] == 2
    assert proposal["proposal"]["target_active_memory_slot_count"] == 1
    assert proposal["proposal"]["avoids_all_slot_scan"] is True
    assert proposal["mutates_runtime_state"] is False
    assert proposal["promotion_gate"]["eligible_for_checkpointed_transaction"] is True
    assert proposal["promotion_gate"]["new_slots_initially_neutral"] is True
    assert report["surface"] == "marulho_language_structural_plasticity_transaction.v1"
    assert report["applied"] is True
    assert report["mutation"]["proposal_kind"] == "memory_slot_expansion"
    assert report["mutation"]["source_memory_slot_count"] == 0
    assert report["mutation"]["target_memory_slot_count"] == 4
    assert report["mutation"]["memory_slot_count_delta"] == 4
    assert report["mutation"]["target_memory_slot_candidate_count"] == 2
    assert report["mutation"]["target_active_memory_slot_count"] == 1
    assert memory_model.config.memory_slot_count == 4
    assert memory_model.config.memory_slot_candidate_count == 2
    assert memory_model.config.active_memory_slot_count == 1
    torch.testing.assert_close(expanded_logits, baseline_logits)
    assert memory_model.memory_slots is not None
    assert torch.count_nonzero(memory_model.memory_slots).item() > 0
    assert memory_model.memory_slot_gate is not None
    assert memory_model.memory_slot_gate.detach().item() == 0.0
    assert memory_model.memory_slot_gate.grad is not None
    assert memory_model.memory_slot_gate.grad.detach().abs().item() > 0.0
    assert memory["enabled"] is True
    assert memory["total_slots"] == 4
    assert memory["candidate_slot_count"] == 2
    assert memory["active_slots_per_token"] == 1
    assert memory["candidate_slots_scored"] == split.eval[0].input_ids.numel() * 2
    assert memory["runs_all_slots"] is False
    assert memory["memory_gate_readback"] is False
    assert metadata["transaction"] == "memory_slot_expansion"
    assert restored_model.config.memory_slot_count == 4
    assert restored_model.config.memory_slot_candidate_count == 2
    assert restored_model.memory_slots is not None
    assert report["checkpoint"]["checkpoint_restore_verified"] is True
    assert report["rollback_evidence"]["rollback_verified"] is True
    assert report["promotion_gate"][
        "eligible_for_reviewed_memory_slot_expansion_promotion"
    ] is True
    assert report["promotion_gate"]["eligible_for_reviewed_growth_promotion"] is False


def test_memory_slot_hot_training_skips_per_step_stats_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=8)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=16,
            memory_slot_count=4,
            memory_slot_candidate_count=2,
            active_memory_slot_count=1,
        )
    )
    calls = 0
    original = language_model_module.language_memory_slots_triton_stats_delta

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        language_model_module,
        "language_memory_slots_triton_stats_delta",
        wrapped,
    )

    hot_result = model.next_token_loss(
        split.train[0].input_ids,
        split.train[0].target_ids,
        collect_telemetry=False,
        return_evidence=False,
    )
    hot_result["loss"].backward()

    assert calls == 0
    assert "telemetry" not in hot_result

    evidence_result = model.next_token_loss(
        split.train[0].input_ids,
        split.train[0].target_ids,
        collect_telemetry=False,
        return_evidence=True,
    )
    memory = evidence_result["telemetry"]["memory"]

    assert calls == 1
    assert memory["evidence_collected"] is True
    assert memory["memory_slot_triton_stats_delta"]["surface"] == (
        "marulho_language_memory_slots_triton_stats_delta.v1"
    )
    assert memory["memory_slot_retrieval_backend"] == (
        "torch_autograd_bounded_memory_slots"
    )


def test_language_checkpoint_evolution_forks_child_without_mutating_parent(tmp_path) -> None:
    torch.manual_seed(29)
    tokenizer = ByteLevelLanguageTokenizer()
    parent_split = build_language_model_splits(
        ["parent checkpoint evidence protects old language state. " * 7],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    child_split = build_language_model_splits(
        ["child evolution learns a separate replay protected domain. " * 7],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    model.train()

    child, report = run_language_checkpoint_evolution(
        model,
        tokenizer,
        eval_batches=parent_split.eval,
        child_train_batches=child_split.train[:2],
        child_new_eval_batches=child_split.train[:2],
        replay_batches=parent_split.train[:1],
        checkpoint_dir=tmp_path,
        config=LanguageCheckpointEvolutionConfig(
            max_child_loss_delta=100.0,
            max_old_domain_forgetting=100.0,
            require_child_learning=False,
            allow_structural_growth=True,
        ),
        learning_config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=2,
            replay_loss_weight=0.25,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
            rollback_on_forgetting=False,
        ),
        structural_config=LanguageStructuralPlasticityConfig(
            route_saturation_threshold=0.0,
            max_eval_loss_delta=100.0,
        ),
    )

    lineage = report["lineage"]
    checkpoint_lineage = report["checkpoint_lineage"]
    runtime_evidence = report["runtime_evidence"]
    review = report["evolution_review"]
    gate = report["promotion_gate"]
    assert report["surface"] == "marulho_language_checkpoint_evolution.v1"
    assert report["mutates_parent_runtime"] is False
    assert report["external_llm_used"] is False
    assert model.training is True
    assert lineage["parent_state_hash_before"] == lineage["parent_state_hash_after"]
    assert lineage["parent_training_mode_before"] is True
    assert lineage["parent_training_mode_after"] is True
    assert lineage["parent_state_hash_before"] == lineage["child_state_hash_initial"]
    assert lineage["parent_state_hash_before"] != lineage["child_state_hash_final"]
    assert Path(lineage["parent_checkpoint"]).exists()
    assert Path(lineage["child_initial_checkpoint"]).exists()
    assert Path(lineage["child_final_checkpoint"]).exists()
    assert checkpoint_lineage["surface"] == (
        "marulho_language_checkpoint_evolution_lineage.v1"
    )
    assert checkpoint_lineage["lineage_complete"] is True
    assert checkpoint_lineage["child_initial_matches_parent_state"] is True
    assert checkpoint_lineage["child_final_matches_child_runtime"] is True
    assert checkpoint_lineage["child_final_differs_from_parent_state"] is True
    assert checkpoint_lineage["mutates_parent_checkpoint"] is False
    assert checkpoint_lineage["parent_checkpoint_sha256"]
    assert checkpoint_lineage["child_initial_checkpoint_sha256"]
    assert checkpoint_lineage["child_final_checkpoint_sha256"]
    assert runtime_evidence["surface"] == (
        "marulho_language_checkpoint_evolution_runtime_truth.v1"
    )
    assert runtime_evidence["parent_model_device"] == str(model.device)
    assert runtime_evidence["child_model_device"] == str(child.device)
    assert runtime_evidence["checkpoint_storage_device"] == "cpu"
    assert runtime_evidence["child_update_token_count"] > 0
    assert runtime_evidence["child_optimizer_step_count"] > 0
    assert runtime_evidence["per_step_metric_cpu_sync"] is False
    assert review["surface"] == "marulho_language_checkpoint_evolution_review.v1"
    assert review["lineage_complete"] is True
    assert review["isolated_child_training"] is True
    assert review["parent_kept_installed"] is True
    assert (
        review["child_update_token_count"]
        == runtime_evidence["child_update_token_count"]
    )
    assert review["structural_growth_attempted"] is True
    assert review["structural_checkpoint_backed"] is True
    assert review["operator_review_required"] is True
    assert review["long_run_evidence_required_for_promotion"] is True
    assert review["promotion_mutates_parent_runtime"] is False
    assert report["comparison"]["parent_rollback_verified"] is True
    assert report["comparison"]["parent_training_mode_unchanged"] is True
    assert gate["parent_runtime_unchanged"] is True
    assert gate["rollback_to_parent_verified"] is True
    assert gate["child_checkpoint_available"] is True
    assert gate["checkpoint_lineage_complete"] is True
    assert gate["long_run_evidence_required_for_parent_promotion"] is True
    assert report["structural_transaction"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert child.config.expert_count >= model.config.expert_count


def test_language_eval_generation_and_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(11)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(_texts(), tokenizer, sequence_length=14)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for batch in split.train[:3]:
        optimizer.zero_grad(set_to_none=True)
        result = model.next_token_loss(batch.input_ids, batch.target_ids)
        result["loss"].backward()
        optimizer.step()

    report = evaluate_language_model(model, split.eval)
    assert model.training is True
    prompt = torch.tensor(tokenizer.encode("marulho", add_eos=False), dtype=torch.long)
    generation = model.generate(prompt, max_new_tokens=3, eos_id=tokenizer.eos_id)
    assert model.training is True
    checkpoint_path = save_language_model_checkpoint(
        tmp_path / "language-model.pt",
        model,
        tokenizer,
        metadata={"train_split_hash": split.report["train_split_hash"]},
    )
    restored_model, restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path
    )
    restored_model.eval()
    model.eval()

    with torch.no_grad():
        original_logits = model(split.eval[0].input_ids)["logits"]
        restored_logits = restored_model(split.eval[0].input_ids)["logits"]

    assert report["surface"] == "marulho_language_model_heldout_evaluation.v1"
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["external_llm_used"] is False
    assert math.isfinite(report["heldout_loss"])
    assert report["heldout_perplexity"] > 0
    assert report["metric_readback_mode"] == "deferred_gpu_scalar_aggregation"
    assert report["per_batch_metric_cpu_sync"] is False
    assert report["evidence_collection_mode"] == "last_batch_only"
    assert report["per_batch_evidence_dict_build"] is False
    assert report["evidence_probe_batch_tokens"] > 0
    assert report["caller_device_transfer_calls"] == 0
    assert report["elapsed_seconds"] >= 0.0
    assert report["tokens_per_second"] > 0.0
    assert generation["surface"] == "marulho_language_generation.v1"
    assert generation["active_language_path"] == "marulho_lm_head"
    assert generation["external_llm_used"] is False
    assert generation["generated_ids"].shape[1] >= prompt.numel()
    assert checkpoint_path.exists()
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata["train_split_hash"] == split.report["train_split_hash"]
    torch.testing.assert_close(original_logits, restored_logits)


def test_language_continual_learning_window_measures_forgetting_and_replay() -> None:
    torch.manual_seed(17)
    tokenizer = ByteLevelLanguageTokenizer()
    old_split = build_language_model_splits(
        ["old domain runtime truth protects replay evidence. " * 8],
        tokenizer,
        sequence_length=12,
        eval_fraction=0.25,
    )
    new_split = build_language_model_splits(
        ["new domain language learning updates checkpointed weights. " * 8],
        tokenizer,
        sequence_length=12,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=24,
        )
    )

    report = run_language_continual_learning_window(
        model,
        new_batches=new_split.train[:2],
        old_eval_batches=old_split.eval,
        new_eval_batches=new_split.train[:2],
        replay_batches=old_split.train[:1],
        config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=4,
            replay_loss_weight=0.25,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
        ),
    )

    assert report["surface"] == "marulho_language_continual_learning_window.v1"
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["mutates_language_model_weights"] is True
    assert report["learning_evidence"]["new_domain_loss_delta"] > 0.0
    assert report["learning_evidence"]["final_parameter_delta_l2"] > 0.0
    assert report["learning_evidence"]["optimizer_policy"] == "AdamW_all_parameters"
    assert report["learning_evidence"]["dense_adamw_backend"] == "default"
    assert report["learning_evidence"]["optimizer_step_count"] == 8
    fusion = report["learning_evidence"]["paired_update_replay_fusion"]
    assert fusion["surface"] == (
        "marulho_language_continual_paired_update_replay_fusion.v1"
    )
    assert fusion["enabled"] is True
    assert fusion["mode"] == "single_hidden_forward_split_update_replay_losses"
    assert fusion["weighted_replay_loss_preserved"] is True
    assert fusion["actual_fused_steps"] == 8
    assert fusion["separate_replay_forward_loss_calls_avoided"] == 8
    assert report["learning_evidence"]["measured_update_loop_model_loss_calls"] == 8
    assert report["learning_evidence"]["gradient_clip_mode"] == (
        "sparse_aware_device_norm_every_step"
    )
    assert report["learning_evidence"]["gradient_clip_applied_step_count"] == 8
    assert report["learning_evidence"]["gradient_clip_skipped_step_count"] == 0
    assert report["learning_evidence"]["metric_readback_mode"] == (
        "deferred_gpu_scalar_aggregation"
    )
    assert report["learning_evidence"]["per_step_metric_cpu_sync"] is False
    batch_device_staging = report["learning_evidence"]["batch_device_staging"]
    assert batch_device_staging["surface"] == (
        "marulho_language_continual_batch_device_staging.v1"
    )
    assert batch_device_staging["staged_before_measured_update_window"] is True
    assert batch_device_staging["all_update_batches_on_device_before_timing"] is True
    assert batch_device_staging[
        "measured_update_loop_caller_device_transfer_calls"
    ] == 0
    assert report["learning_evidence"][
        "measured_update_loop_caller_device_transfer_calls"
    ] == 0
    assert report["learning_evidence"][
        "training_window_memory_slot_triton_stats_delta"
    ]["surface"] == "marulho_language_memory_slots_triton_stats_delta.v1"
    assert report["learning_evidence"][
        "training_window_memory_slot_triton_autograd_used"
    ] is False
    triton_accounting = report["learning_evidence"][
        "training_window_triton_accounting"
    ]
    assert triton_accounting["surface"] == (
        "marulho_language_continual_training_window_triton_accounting.v1"
    )
    assert triton_accounting["scope"] == "measured_update_window_only"
    assert triton_accounting["tracked_kernel_names"] == [
        "language_rmsnorm_triton",
        "language_plif_triton",
        "language_route_topk_triton",
        "language_expert_dispatch_triton",
        "language_memory_slots_triton",
        "language_sampled_vocab_ce_triton",
    ]
    assert set(triton_accounting["tracked_kernel_used_names"]).issubset(
        set(triton_accounting["tracked_kernel_names"])
    )
    assert isinstance(triton_accounting["tracked_torch_fallback_calls"], int)
    assert isinstance(triton_accounting["tracked_triton_failure_count"], int)
    assert triton_accounting["language_memory_slots_triton"]["surface"] == (
        "marulho_language_memory_slots_triton_stats_delta.v1"
    )
    assert report["learning_evidence"]["window_phase_timings"]["surface"] == (
        "marulho_language_continual_window_phase_timings.v1"
    )
    assert "batch_device_staging_seconds" in report["learning_evidence"][
        "window_phase_timings"
    ]
    assert report["learning_evidence"]["window_phase_timings"][
        "total_window_seconds"
    ] >= report["learning_evidence"]["window_phase_timings"]["update_seconds"]
    assert report["learning_evidence"]["total_window_tokens_per_second"] > 0.0
    assert report["learning_evidence"]["sampled_vocab_training"] is False
    assert report["learning_evidence"]["sampled_vocab_precompute"]["new_batches"][
        "enabled"
    ] is False
    assert report["learning_evidence"]["sampled_vocab_precompute"]["new_batches"][
        "reason"
    ] == "sampled_vocab_training_disabled"
    for eval_report in (
        report["old_domain_before"],
        report["old_domain_after"],
        report["new_domain_before"],
        report["new_domain_after"],
        report["replay_before"],
        report["replay_after"],
    ):
        assert eval_report["metric_readback_mode"] == (
            "deferred_gpu_scalar_aggregation"
        )
        assert eval_report["per_batch_metric_cpu_sync"] is False
        assert eval_report["evidence_collection_mode"] == "last_batch_only"
        assert eval_report["per_batch_evidence_dict_build"] is False
        assert eval_report["evidence_probe_batch_tokens"] > 0
        assert eval_report["elapsed_seconds"] >= 0.0
        assert eval_report["tokens_per_second"] > 0.0
    assert "old_domain_forgetting" in report["learning_evidence"]
    assert "general_replay_retention_delta" in report["learning_evidence"]
    assert report["rollback_evidence"]["rollback_applied"] is False
    assert report["rollback_evidence"]["restore_verified"] is True
    assert report["promotion_gate"]["old_domain_forgetting_within_tolerance"] is True
    assert model.training is True


def test_language_continual_learning_supports_sampled_padded_vocab_sparse_updates() -> None:
    torch.manual_seed(20260704)
    tokenizer = ByteLevelLanguageTokenizer()
    old_split = build_language_model_splits(
        ["old replay domain protects retained language evidence. " * 8],
        tokenizer,
        sequence_length=12,
        eval_fraction=0.25,
        batch_size=2,
    )
    new_split = build_language_model_splits(
        ["new sampled vocabulary domain updates sparse rows. " * 8],
        tokenizer,
        sequence_length=12,
        eval_fraction=0.25,
        batch_size=2,
    )
    model_vocab_size = tokenizer.vocab_size + 128
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=16,
            state_dim=24,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=32,
            sampled_vocab_size=32,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
            generation_vocab_size=tokenizer.vocab_size,
            recurrent_gradient_horizon=4,
        )
    )

    report = run_language_continual_learning_window(
        model,
        new_batches=new_split.train[:2],
        old_eval_batches=old_split.eval,
        new_eval_batches=new_split.train[:2],
        replay_batches=old_split.train[:1],
        config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=2,
            replay_loss_weight=0.25,
            gradient_clip_interval=2,
            dense_adamw_backend="foreach",
            paired_sampled_vocab_loss=True,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
            rollback_on_forgetting=False,
        ),
    )

    evidence = report["learning_evidence"]
    assert report["surface"] == "marulho_language_continual_learning_window.v1"
    assert report["model_vocab_size"] == model_vocab_size
    assert report["generation_vocab_size"] == tokenizer.vocab_size
    assert report["sampled_vocab_size"] == 32
    assert report["sparse_vocab_optimizer"] is True
    assert evidence["optimizer_policy"] == (
        "AdamW_foreach_dense_core_plus_SparseAdam_vocab_rows"
    )
    assert evidence["dense_adamw_backend"] == "foreach"
    assert evidence["optimizer_step_count"] == 4
    fusion = evidence["paired_update_replay_fusion"]
    assert fusion["enabled"] is True
    assert fusion["actual_fused_steps"] == 4
    assert fusion["separate_replay_forward_loss_calls_avoided"] == 4
    assert evidence["measured_update_loop_model_loss_calls"] == 4
    assert fusion["paired_sampled_vocab_loss_fused_steps"] == 4
    assert fusion["sampled_vocab_ce_loss_calls_avoided"] == 4
    paired_sampled_vocab = evidence["sampled_vocab_precompute"][
        "paired_update_replay_batches"
    ]
    assert paired_sampled_vocab["enabled"] is True
    assert paired_sampled_vocab["pair_count"] == 2
    assert paired_sampled_vocab["hot_update_window_precomputed"] is True
    assert evidence["gradient_clip_mode"] == (
        "sparse_aware_device_norm_every_n_steps"
    )
    assert evidence["gradient_clip_interval"] == 2
    assert evidence["gradient_clip_applied_step_count"] == 2
    assert evidence["gradient_clip_skipped_step_count"] == 2
    assert evidence["metric_readback_mode"] == "deferred_gpu_scalar_aggregation"
    assert evidence["per_step_metric_cpu_sync"] is False
    assert evidence["memory_slot_hot_update_evidence_mode"] == (
        "training_window_counter_delta_plus_post_window_probe"
    )
    assert evidence["per_step_memory_slot_stats_delta"] is False
    assert evidence["training_window_memory_slot_triton_stats_delta"]["surface"] == (
        "marulho_language_memory_slots_triton_stats_delta.v1"
    )
    assert evidence["training_window_memory_slot_triton_autograd_used"] is False
    triton_accounting = evidence["training_window_triton_accounting"]
    assert "language_sampled_vocab_ce_triton" in triton_accounting[
        "tracked_kernel_names"
    ]
    assert triton_accounting["language_sampled_vocab_ce_triton"]["surface"] == (
        "marulho_language_sampled_vocab_ce_triton_stats_delta.v1"
    )
    assert isinstance(
        triton_accounting["tracked_triton_autograd_forward_calls"],
        int,
    )
    assert evidence["sampled_vocab_training"] is True
    assert evidence["full_vocab_logits_materialized"] is False
    assert evidence["sampled_vocab_precompute"]["surface"] == (
        "marulho_language_continual_sampled_vocab_precompute.v1"
    )
    assert evidence["sampled_vocab_precompute"]["old_eval_batches"]["enabled"] is True
    assert evidence["sampled_vocab_precompute"]["new_eval_batches"]["enabled"] is True
    assert evidence["sampled_vocab_precompute"]["new_batches"]["enabled"] is True
    assert evidence["sampled_vocab_precompute"]["new_batches"]["batch_count"] == 2
    assert evidence["sampled_vocab_precompute"]["replay_batches"]["enabled"] is True
    assert evidence["sampled_vocab_precompute"]["replay_batches"]["batch_count"] == 1
    phase_timings = evidence["window_phase_timings"]
    assert phase_timings["surface"] == (
        "marulho_language_continual_window_phase_timings.v1"
    )
    assert phase_timings["sampled_vocab_precompute_seconds"] >= 0.0
    assert phase_timings["pre_update_evaluation_seconds"] >= 0.0
    assert phase_timings["post_update_evaluation_seconds"] >= 0.0
    assert phase_timings["total_window_seconds"] >= phase_timings["update_seconds"]
    assert evidence["total_window_elapsed_seconds"] == phase_timings[
        "total_window_seconds"
    ]
    assert evidence["total_window_tokens_per_second"] > 0.0
    assert evidence["loss_evidence"]["sampled_vocab_id_source"] == (
        "precomputed_batch_sampled_vocab_ids"
    )
    assert evidence["loss_evidence"]["sampled_target_position_source"] == (
        "precomputed_batch_target_positions"
    )
    assert evidence["loss_evidence"]["precomputed_sampled_vocab_used"] is True
    assert evidence["loss_evidence"]["precomputed_target_positions_used"] is True
    assert evidence["replay_loss_evidence"]["precomputed_sampled_vocab_used"] is True
    assert evidence["replay_loss_evidence"]["precomputed_target_positions_used"] is True
    assert report["old_domain_before"]["spike_telemetry"]["vocab_loss"][
        "precomputed_sampled_vocab_used"
    ] is True
    assert report["new_domain_after"]["spike_telemetry"]["vocab_loss"][
        "precomputed_sampled_vocab_used"
    ] is True
    assert report["replay_after"]["spike_telemetry"]["vocab_loss"][
        "precomputed_sampled_vocab_used"
    ] is True
    for eval_report in (
        report["old_domain_before"],
        report["old_domain_after"],
        report["new_domain_before"],
        report["new_domain_after"],
        report["replay_before"],
        report["replay_after"],
    ):
        assert eval_report["metric_readback_mode"] == (
            "deferred_gpu_scalar_aggregation"
        )
        assert eval_report["per_batch_metric_cpu_sync"] is False
        assert eval_report["elapsed_seconds"] >= 0.0
        assert eval_report["tokens_per_second"] > 0.0
        assert eval_report["spike_telemetry"]["vocab_loss"][
            "precomputed_sampled_vocab_used"
        ] is True
    assert evidence["loss_evidence"]["lm_head_weight_gradient_sparse"] is True
    assert evidence["loss_evidence"]["token_embedding_gradient_sparse"] is True
    assert evidence["final_parameter_delta_l2"] > 0.0
    assert report["rollback_evidence"]["rollback_applied"] is False

    guard_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=16,
            state_dim=24,
            sampled_vocab_size=32,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
            generation_vocab_size=tokenizer.vocab_size,
        )
    )
    with pytest.raises(ValueError, match="sparse_vocab_optimizer"):
        run_language_continual_learning_window(
            guard_model,
            new_batches=new_split.train[:1],
            old_eval_batches=old_split.eval,
            new_eval_batches=new_split.train[:1],
            config=LanguageContinualLearningConfig(
                sparse_vocab_optimizer=False,
                forgetting_tolerance=100.0,
                replay_retention_tolerance=100.0,
            ),
        )
