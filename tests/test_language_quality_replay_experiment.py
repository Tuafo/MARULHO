from __future__ import annotations

import json
from pathlib import Path

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
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=64,
            ),
        ),
        heldout_prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="Replay protects",
                source_text=source_text,
                max_new_tokens=4,
                min_new_tokens=1,
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=64,
            ),
            LanguageGenerationPromptCase(
                prompt_text="Long sustained runs",
                source_text=source_text,
                max_new_tokens=4,
                min_new_tokens=1,
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
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
            heldout_prompt_case_count=2,
            sustained_target_token_counts=(2, 3),
            sustained_tick_tokens=1,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            benchmark_suite_output_path=str(tmp_path / "quality-replay-suite.json"),
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
    assert (tmp_path / "quality-replay-parent-heldout-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-heldout-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-sustained-2.json").exists()
    assert (tmp_path / "quality-replay-child-sustained-3.json").exists()
    assert (tmp_path / "quality-replay-suite.json").exists()
    assert report["hard_prompt_replay"]["source_prompt_found_count"] == 1
    assert report["split"]["used_new_batches"] == 1
    assert report["split"]["used_replay_batches"] == 1
    assert report["learning_evidence"]["learning_evidence"]["update_token_count"] > 0
    assert report["candidate_selection"]["candidate_count"] == 1
    assert report["candidate_selection"]["selected_candidate_id"] == "candidate-00"
    assert report["candidate_selection"]["mutates_parent_checkpoint"] is False
    assert report["candidate_selection"][
        "runs_sustained_runtime_only_for_selected_child"
    ] is True
    assert len(report["candidate_selection"]["candidates"]) == 1
    assert report["candidate_selection"]["candidates"][0]["selected"] is True
    assert report["generation_coherence_after"]["checkpoint_path"] == str(child)
    assert report["generation_coherence_delta"]["surface"] == (
        "marulho_language_quality_replay_coherence_delta.v1"
    )
    assert report["heldout_prompt_suite"]["enabled"] is True
    assert report["heldout_prompt_suite"]["case_count"] == 2
    assert report["heldout_prompt_suite"]["source"] == "explicit_heldout_prompt_cases"
    assert report["heldout_prompt_suite"]["not_used_for_replay_training"] is True
    assert report["heldout_generation_coherence_before"]["checkpoint_path"] == str(parent)
    assert report["heldout_generation_coherence_after"]["checkpoint_path"] == str(child)
    assert report["heldout_generation_coherence_delta"]["surface"] == (
        "marulho_language_quality_replay_coherence_delta.v1"
    )
    assert report["quality_generalization_review"]["surface"] == (
        "marulho_language_quality_replay_generalization_review.v1"
    )
    assert report["quality_generalization_review"][
        "heldout_prompt_coherence_recorded"
    ] is True
    assert report["sustained_runtime_evidence_summary"]["all_success"] is True
    assert report["sustained_runtime_evidence_summary"]["report_count"] == 2
    assert report["sustained_runtime_evidence_summary"]["target_tokens"] == [2, 3]
    assert len(report["sustained_runtime_evidence_reports"]) == 2
    assert {
        item["checkpoint_path"] for item in report["sustained_runtime_evidence_reports"]
    } == {str(child)}
    assert report["benchmark_suite_report"]["surface"] == (
        "marulho_language_runtime_benchmark_suite.v1"
    )
    assert report["benchmark_suite_report"]["promotion_gate"][
        "quality_replay_evidence_available"
    ] is True
    suite_categories = {
        item["name"]: item for item in report["benchmark_suite_report"]["categories"]
    }
    suite_quality_replay = suite_categories["generation_coherence"]["evidence"][
        "quality_replay_evidence"
    ]
    assert suite_quality_replay["quality_replay_available"] is True
    assert suite_quality_replay["best_report"]["selected_candidate_id"] == (
        "candidate-00"
    )
    assert suite_quality_replay["best_report"]["selected_child_checkpoint_path"] == (
        str(child)
    )
    assert report["experiment_review"]["records_hard_prompt_training_pressure"] is True
    assert report["experiment_review"]["records_candidate_child_selection"] is True
    assert report["experiment_review"]["candidate_count"] == 1
    assert report["experiment_review"]["records_same_child_generation_coherence"] is True
    assert report["experiment_review"]["records_heldout_generation_coherence"] is True
    assert report["experiment_review"]["records_same_child_sustained_runtime"] is True
    assert report["experiment_review"]["records_multiple_sustained_targets"] is True
    assert report["experiment_review"]["records_benchmark_suite_aggregation"] is True
    assert report["experiment_review"][
        "benchmark_suite_quality_replay_evidence_available"
    ] is True
    assert report["experiment_review"]["promotes_generation_quality_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_language_quality_replay_experiment_selects_from_multiple_child_candidates(
    tmp_path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    source_text = (
        "MARULHO learns runtime evidence from local source windows. "
        "Replay protects old evidence while new source text updates the LM head. "
        "Structural pressure can grow, prune, merge, or sleep experts under review. "
    ) * 2
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

    output = tmp_path / "quality-replay-sweep.json"
    report = run_language_quality_replay_experiment(
        checkpoint_path=parent,
        output_path=output,
        source_path=source,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="MARULHO",
                source_text=source_text,
                max_new_tokens=3,
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
            hard_prompt_repeat=1,
            hard_prompt_context_chars=80,
            max_new_batches=1,
            max_replay_batches=1,
            max_old_eval_batches=1,
            max_new_eval_batches=1,
            max_steps=1,
            learning_rate=1e-3,
            replay_loss_weight=0.25,
            candidate_learning_rates=(1e-3, 5e-4),
            candidate_replay_loss_weights=(0.25, 0.75),
            candidate_max_steps=(1, 1),
            gradient_clip_interval=1,
            generation_repetition_penalty=1.0,
            generation_no_repeat_ngram_size=0,
            heldout_prompt_case_count=2,
            device="cpu",
        ),
    )

    selection = report["candidate_selection"]
    selected = selection["selected_candidate_id"]
    selected_candidates = [
        candidate for candidate in selection["candidates"] if candidate["selected"]
    ]
    assert selection["enabled"] is True
    assert selection["candidate_count"] == 2
    assert len(selection["candidates"]) == 2
    assert len(selected_candidates) == 1
    assert selected in {"candidate-00", "candidate-01"}
    assert Path(report["child_checkpoint_path"]).exists()
    assert Path(report["child_checkpoint_path"]).name.startswith(
        f"quality-replay-sweep-{selected}-child-checkpoint"
    )
    assert (tmp_path / "quality-replay-sweep-candidate-00-child-checkpoint.pt").exists()
    assert (tmp_path / "quality-replay-sweep-candidate-01-child-checkpoint.pt").exists()
    assert report["checkpoint_lineage"]["candidate_count"] == 2
    assert report["experiment_review"]["candidate_count"] == 2
