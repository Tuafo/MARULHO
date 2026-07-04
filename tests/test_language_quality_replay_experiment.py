from __future__ import annotations

import json

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_generation_coherence import LanguageGenerationPromptCase
from marulho.evaluation.language_quality_replay_experiment import (
    SURFACE,
    LanguageQualityReplayExperimentConfig,
    run_language_quality_replay_experiment,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def test_language_quality_replay_experiment_writes_child_quality_and_speed_evidence(
    tmp_path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    source_text = (
        "MARULHO learns runtime evidence from local source windows. "
        "Replay protects old evidence while new source text updates the LM head. "
        "Structural pressure can grow, prune, merge, or sleep experts under review. "
        "Long sustained runs measure token throughput and fallback truth. "
    ) * 3
    source = tmp_path / "source.txt"
    source.write_text(source_text, encoding="utf-8")
    parent = tmp_path / "parent.pt"
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=384,
            embedding_dim=8,
            state_dim=12,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=16,
            sampled_vocab_size=32,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
            generation_vocab_size=tokenizer.vocab_size,
            recurrent_gradient_horizon=2,
        )
    )
    save_language_model_checkpoint(parent, model, tokenizer)

    output = tmp_path / "quality-replay.json"
    report = run_language_quality_replay_experiment(
        checkpoint_path=parent,
        output_path=output,
        source_path=source,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="MARULHO",
                source_text=source_text,
                max_new_tokens=4,
                min_new_tokens=1,
                min_prefix_match_chars=1,
                min_prefix_match_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=64,
            ),
        ),
        config=LanguageQualityReplayExperimentConfig(
            sequence_length=8,
            stride=4,
            batch_size=2,
            hard_prompt_repeat=2,
            hard_prompt_context_chars=80,
            max_new_batches=1,
            max_replay_batches=1,
            max_old_eval_batches=1,
            max_new_eval_batches=1,
            max_steps=1,
            learning_rate=1e-3,
            replay_loss_weight=0.25,
            gradient_clip_interval=1,
            generation_repetition_penalty=1.0,
            generation_no_repeat_ngram_size=0,
            sustained_target_tokens=3,
            sustained_tick_tokens=1,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            device="cpu",
        ),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))
    child = tmp_path / "quality-replay-child-checkpoint.pt"

    assert output.exists()
    assert loaded["surface"] == SURFACE
    assert report["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["checkpoint_lineage"]["writes_child_checkpoint"] is True
    assert report["checkpoint_lineage"]["mutates_parent_checkpoint"] is False
    assert child.exists()
    assert (tmp_path / "quality-replay-parent-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-sustained-3.json").exists()
    assert report["hard_prompt_replay"]["source_prompt_found_count"] == 1
    assert report["split"]["used_new_batches"] == 1
    assert report["split"]["used_replay_batches"] == 1
    assert report["learning_evidence"]["learning_evidence"]["update_token_count"] > 0
    assert report["generation_coherence_after"]["checkpoint_path"] == str(child)
    assert report["generation_coherence_delta"]["surface"] == (
        "marulho_language_quality_replay_coherence_delta.v1"
    )
    assert report["sustained_runtime_evidence"]["success"] is True
    assert report["sustained_runtime_evidence"]["checkpoint_path"] == str(child)
    assert report["experiment_review"]["records_hard_prompt_training_pressure"] is True
    assert report["experiment_review"]["records_same_child_generation_coherence"] is True
    assert report["experiment_review"]["records_same_child_sustained_runtime"] is True
    assert report["experiment_review"]["promotes_generation_quality_claim"] is False
    assert (tmp_path / "README.md").exists()
