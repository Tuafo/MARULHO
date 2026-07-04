from __future__ import annotations

from marulho.evaluation.language_sampled_vocab_training_impact import (
    SampledVocabTrainingImpactConfig,
    run_language_sampled_vocab_training_impact,
)


def test_language_sampled_vocab_training_impact_reports_full_step(tmp_path) -> None:
    output = tmp_path / "sampled-vocab-training-impact.json"

    report = run_language_sampled_vocab_training_impact(
        output_path=output,
        config=SampledVocabTrainingImpactConfig(
            vocab_size=384,
            sampled_vocab_size=32,
            embedding_dim=12,
            state_dim=16,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=24,
            sequence_length=8,
            batch_size=2,
            warmup_steps=0,
            repeats=1,
            device="cpu",
        ),
    )

    assert output.exists()
    assert report["surface"] == "marulho_language_sampled_vocab_training_impact.v1"
    assert report["model_vocab_size"] == 384
    assert report["tokenizer_vocab_size"] < report["model_vocab_size"]
    assert report["review"]["complete_training_step_impact"] is True
    assert report["review"]["not_kernel_microbench_only"] is True
    dense = report["arms"]["dense_full_vocab"]
    sampled = report["arms"]["sampled_adaptive_vocab"]
    assert dense["success"] is True
    assert sampled["success"] is True
    assert dense["full_vocab_logits_materialized"] is True
    assert sampled["full_vocab_logits_materialized"] is False
    assert sampled["sampled_vocab_training"] is True
    assert sampled["optimizer_policy"] == "AdamW_dense_core_plus_SparseAdam_vocab_rows"
    assert sampled["loss_evidence"]["loss_backend"] == (
        "torch_autograd_selected_lm_head_rows"
    )
    assert sampled["loss_evidence"]["lm_head_weight_gradient_sparse"] is True
    assert sampled["loss_evidence"]["token_embedding_gradient_sparse"] is True
    assert (
        sampled["sampled_vocab_ce_triton_stats_delta"]["triton_kernel_used"]
        is False
    )
    assert sampled["token_count"] == report["batch"]["tokens_per_optimizer_step"]
    assert report["comparison"]["sampled_training_success"] is True
    assert report["comparison"]["scalability_evidence"] == "sampled_vs_dense_measured"
