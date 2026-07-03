from __future__ import annotations

import math
from pathlib import Path

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_plasticity_proposal,
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
    assert result["telemetry"]["adaptive_timestep_budget"] == 2
    assert result["telemetry"]["adaptive_step_count"] == split.train[0].input_ids.shape[1] * 2
    assert result["telemetry"]["input_dependent_leak"] is True
    assert result["telemetry"]["input_dependent_threshold"] is True
    assert result["telemetry"]["trainable_current_terms"] is True
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
    assert routing["total_columns"] == 6
    assert 1 <= routing["active_columns"] <= 6
    assert routing["active_expert_count_per_token"] == 2
    assert routing["route_candidate_count"] == 3
    assert routing["candidate_rows_scored"] == split.train[0].input_ids.numel() * 3
    assert routing["output_candidate_count"] == 2
    assert routing["runs_all_columns"] is False
    assert routing["fallback_reason"] is None
    assert routing["route_device"] == "cpu"
    assert routing["route_latency_ms"] >= 0.0
    assert routing["active_parameters_per_token"] > 0
    assert model.routed_experts.route_keys.grad is not None
    assert torch.isfinite(model.routed_experts.route_keys.grad).all()


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
    prompt = torch.tensor(tokenizer.encode("marulho", add_eos=False), dtype=torch.long)
    generation = model.generate(prompt, max_new_tokens=3, eos_id=tokenizer.eos_id)
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
    assert "old_domain_forgetting" in report["learning_evidence"]
    assert "general_replay_retention_delta" in report["learning_evidence"]
    assert report["rollback_evidence"]["rollback_applied"] is False
    assert report["rollback_evidence"]["restore_verified"] is True
    assert report["promotion_gate"]["old_domain_forgetting_within_tolerance"] is True
