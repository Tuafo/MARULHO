from __future__ import annotations

import json
from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_generation_coherence import LanguageGenerationPromptCase
from marulho.evaluation.language_quality_replay_experiment import (
    SURFACE,
    LanguageQualityReplayExperimentConfig,
    _candidate_quality_retention_review,
    _candidate_selection_rank,
    _candidate_selection_rank_metrics,
    failed_prompt_cases_from_coherence_report,
    prompt_cases_from_coherence_report,
    run_language_quality_replay_experiment,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def _write_memory_slot_runtime_impact_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_memory_slot_runtime_impact",
                "surface": "marulho_language_memory_slot_runtime_impact.v1",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "model_vocab_size": 524288,
                "batch": {"tokens_per_forward": 1024},
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


def _write_memory_slot_architecture_cost_report(path: Path) -> None:
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
                    "memory_slots": {
                        "enabled": True,
                        "bounded_memory_slot_path": True,
                        "runs_all_slots": False,
                        "candidate_slots_scored": 4194304,
                        "candidate_id_source": "precomputed_batch_memory_candidate_ids",
                        "memory_slot_retrieval_backend": (
                            "torch_autograd_bounded_memory_slots"
                        ),
                    },
                },
                "training_memory_slot_backend_summary": {
                    "training_window_stats_recorded": True,
                    "memory_slot_retrieval_backend": (
                        "torch_autograd_bounded_memory_slots"
                    ),
                    "triton_autograd_used": False,
                },
                "memory_slot_architecture_cost": {
                    "surface": (
                        "marulho_language_continual_memory_slot_architecture_cost.v1"
                    ),
                    "status": "memory_slot_architecture_cost_measured",
                    "comparison_is_no_memory_baseline": True,
                    "comparable_update_throughput": True,
                    "comparable_total_window_throughput": True,
                    "current_update_tokens_per_second": 3753.246,
                    "comparison_update_tokens_per_second": 3765.911,
                    "delta_vs_no_memory_update_percent": -0.336,
                    "current_total_window_tokens_per_second": 3436.735,
                    "comparison_total_window_tokens_per_second": 3451.048,
                    "delta_vs_no_memory_total_window_percent": -0.415,
                    "comparison_report": "no-memory-baseline.json",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_structural_plasticity_transaction_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_structural_plasticity_transaction",
                "surface": "marulho_language_structural_plasticity_transaction.v1",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "active_language_path": "marulho_lm_head",
                "status": "applied_structural_mutation",
                "applied": True,
                "operator_approved": True,
                "checkpoint": {"checkpoint_restore_verified": True},
                "rollback_evidence": {"rollback_verified": True},
                "mutation": {
                    "proposal_kind": "memory_slot_expansion",
                    "source_memory_slot_count": 0,
                    "target_memory_slot_count": 1024,
                    "target_memory_slot_candidate_count": 8,
                    "target_active_memory_slot_count": 2,
                },
                "promotion_gate": {
                    "checkpoint_backed": True,
                    "heldout_non_regression": True,
                    "eligible_for_reviewed_structural_promotion": True,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_failed_prompt_cases_from_coherence_report_imports_only_failures(
    tmp_path,
) -> None:
    source_text = "alpha beta gamma delta epsilon zeta"
    report_path = tmp_path / "coherence.json"
    report_path.write_text(
        json.dumps(
            {
                "surface": "marulho_language_generation_coherence_report.v1",
                "prompt_suite": {
                    "prompt_cases": [
                        {
                            "prompt_text": "alpha beta gamma",
                            "max_new_tokens": 32,
                        },
                        {
                            "prompt_text": "delta epsilon zeta",
                            "max_new_tokens": 48,
                        },
                    ],
                },
                "cases": [
                    {
                        "prompt_text": "alpha beta gamma",
                        "passed": False,
                        "thresholds": {
                            "min_new_tokens": 4,
                            "min_prefix_match_chars": 6,
                            "min_prefix_match_fraction": 0.25,
                            "min_printable_fraction": 0.9,
                            "min_distinct_bigram_fraction": 0.15,
                            "max_token_run_length": 5,
                        },
                    },
                    {
                        "prompt_text": "delta epsilon zeta",
                        "passed": True,
                        "thresholds": {
                            "min_prefix_match_chars": 6,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    all_cases = prompt_cases_from_coherence_report(
        report_path,
        source_text=source_text,
    )
    cases = failed_prompt_cases_from_coherence_report(
        report_path,
        source_text=source_text,
    )

    assert [case.prompt_text for case in all_cases] == [
        "alpha beta gamma",
        "delta epsilon zeta",
    ]
    assert len(cases) == 1
    assert cases[0].prompt_text == "alpha beta gamma"
    assert cases[0].source_text == source_text
    assert cases[0].max_new_tokens == 32
    assert cases[0].min_new_tokens == 4
    assert cases[0].min_prefix_match_chars == 6
    assert cases[0].min_prefix_match_fraction == 0.25
    assert cases[0].min_printable_fraction == 0.9
    assert cases[0].min_distinct_bigram_fraction == 0.15
    assert cases[0].max_token_run_length == 5


def test_candidate_selection_rank_penalizes_passed_loss_regression() -> None:
    learning_report = {
        "status": "accepted_online_update",
        "learning_evidence": {
            "tokens_per_second": 100.0,
            "old_domain_forgetting": 0.0,
            "general_replay_retention_delta": 0.0,
        },
    }
    heldout_delta = {
        "source_continuation_loss_available": True,
        "regressed_prompt_count": 0,
        "passed_case_count_delta": 0,
        "case_pass_rate_delta": 0.0,
        "mean_source_continuation_loss_delta": 0.0,
        "mean_source_continuation_perplexity_delta": 0.0,
    }
    better_loss_delta = {
        "source_continuation_loss_available": True,
        "regressed_prompt_count": 0,
        "passed_case_count_delta": 1,
        "case_pass_rate_delta": 1.0,
        "mean_source_continuation_loss_delta": -0.1,
        "mean_source_continuation_perplexity_delta": -0.2,
    }
    worse_loss_delta = {
        **better_loss_delta,
        "mean_source_continuation_loss_delta": 0.2,
        "mean_source_continuation_perplexity_delta": 0.4,
    }

    better_rank = _candidate_selection_rank(
        learning_report=learning_report,
        trained_delta=better_loss_delta,
        heldout_delta=heldout_delta,
    )
    worse_rank = _candidate_selection_rank(
        learning_report=learning_report,
        trained_delta=worse_loss_delta,
        heldout_delta=heldout_delta,
    )
    better_metrics = _candidate_selection_rank_metrics(better_rank)
    worse_review = _candidate_quality_retention_review(
        trained_delta=worse_loss_delta,
        heldout_delta=heldout_delta,
    )

    assert better_rank > worse_rank
    assert better_metrics["trained_loss_rank"] > 0.0
    assert better_metrics["learning_acceptance"] == 1.0
    assert worse_review["suspicious"] is True
    assert worse_review["suspicious_reasons"] == [
        "trained_prompt_pass_nonregressed_but_loss_regressed"
    ]


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
    memory_slot_runtime_impact = tmp_path / "memory-slot-runtime-impact.json"
    memory_slot_architecture_cost = tmp_path / "memory-slot-architecture-cost.json"
    structural_plasticity = tmp_path / "structural-plasticity-transaction.json"
    _write_memory_slot_runtime_impact_report(memory_slot_runtime_impact)
    _write_memory_slot_architecture_cost_report(memory_slot_architecture_cost)
    _write_structural_plasticity_transaction_report(structural_plasticity)

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
            min_new_loss_improvement=-100.0,
            gradient_clip_interval=1,
            generation_repetition_penalty=1.15,
            generation_no_repeat_ngram_size=2,
            heldout_prompt_case_count=2,
            sustained_target_token_counts=(2, 3),
            sustained_tick_tokens=1,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            benchmark_suite_output_path=str(tmp_path / "quality-replay-suite.json"),
            benchmark_memory_slot_runtime_impact_evidence_paths=(
                str(memory_slot_runtime_impact),
            ),
            benchmark_memory_slot_architecture_cost_evidence_paths=(
                str(memory_slot_architecture_cost),
            ),
            benchmark_structural_plasticity_evidence_paths=(
                str(structural_plasticity),
            ),
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
    assert (tmp_path / "quality-replay-parent-fresh-heldout-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-fresh-heldout-coherence.json").exists()
    assert (tmp_path / "quality-replay-child-sustained-2.json").exists()
    assert (tmp_path / "quality-replay-child-sustained-3.json").exists()
    assert (tmp_path / "quality-replay-suite.json").exists()
    assert report["hard_prompt_replay"]["source_prompt_found_count"] == 1
    assert report["split"]["used_new_batches"] == 1
    assert report["split"]["used_replay_batches"] == 1
    assert report["split"]["old_replay"]["max_train_batches"] == 1
    assert report["split"]["old_replay"]["max_eval_batches"] == 1
    assert report["split"]["hard_prompt"]["max_train_batches"] == 1
    assert report["split"]["hard_prompt"]["max_eval_batches"] == 1
    assert report["split"]["old_replay"]["train_batch_count"] <= 1
    assert report["split"]["hard_prompt"]["train_batch_count"] <= 1
    assert report["learning_evidence"]["learning_evidence"]["update_token_count"] > 0
    assert report["candidate_selection"]["candidate_count"] == 1
    assert report["candidate_selection"]["selected_candidate_id"] == "candidate-00"
    assert report["candidate_selection"]["mutates_parent_checkpoint"] is False
    assert (
        report["candidate_selection"]["heldout_cases_used_for_replay_training"]
        is False
    )
    assert report["candidate_selection"]["heldout_training_prompt_overlap_count"] == 0
    assert report["candidate_selection"]["heldout_training_prompt_overlaps"] == []
    assert (
        report["candidate_selection"]["fresh_heldout_cases_used_for_replay_training"]
        is False
    )
    assert (
        report["candidate_selection"]["fresh_heldout_training_prompt_overlap_count"]
        == 0
    )
    assert report["candidate_selection"]["fresh_heldout_training_prompt_overlaps"] == []
    assert report["candidate_selection"]["fresh_heldout_fixed_prompt_overlap_count"] == 0
    assert report["candidate_selection"]["fresh_heldout_fixed_prompt_overlaps"] == []
    assert report["candidate_selection"][
        "runs_sustained_runtime_only_for_selected_child"
    ] is True
    assert len(report["candidate_selection"]["candidates"]) == 1
    assert report["candidate_selection"]["candidates"][0]["selected"] is True
    assert report["candidate_selection"]["candidates"][0][
        "learning_update_accepted"
    ] is True
    assert report["candidate_selection"]["candidates"][0][
        "selection_rank_learning_acceptance"
    ] == 1.0
    assert "trained_loss_rank" in report["candidate_selection"][
        "selected_selection_rank_metrics"
    ]
    assert report["candidate_selection"]["candidates"][0][
        "selection_rank_metrics"
    ]["learning_acceptance"] == 1.0
    assert report["candidate_selection"]["candidates"][0][
        "quality_retention_review"
    ]["surface"] == "marulho_language_quality_replay_candidate_quality_retention.v1"
    assert report["candidate_selection"]["selected_quality_retention_review"][
        "promotes_generation_quality_claim"
    ] is False
    assert report["generation_coherence_after"]["checkpoint_path"] == str(child)
    assert report["generation_coherence_delta"]["surface"] == (
        "marulho_language_quality_replay_coherence_delta.v1"
    )
    assert report["generation_coherence_delta"][
        "source_continuation_loss_available"
    ] is True
    assert isinstance(
        report["generation_coherence_delta"]["mean_source_continuation_loss_delta"],
        float,
    )
    assert report["heldout_prompt_suite"]["enabled"] is True
    assert report["heldout_prompt_suite"]["case_count"] == 2
    assert report["heldout_prompt_suite"]["source"] == "explicit_heldout_prompt_cases"
    assert report["heldout_prompt_suite"]["not_used_for_replay_training"] is True
    assert report["heldout_prompt_suite"]["training_prompt_overlap_count"] == 0
    assert report["heldout_prompt_suite"]["training_prompt_overlaps"] == []
    assert "source_text" not in report["heldout_prompt_suite"]["prompt_cases"][0]
    assert report["heldout_prompt_suite"]["prompt_cases"][0][
        "raw_source_text_retained"
    ] is False
    assert report["heldout_generation_coherence_before"]["checkpoint_path"] == str(parent)
    assert report["heldout_generation_coherence_after"]["checkpoint_path"] == str(child)
    assert report["heldout_generation_coherence_delta"]["surface"] == (
        "marulho_language_quality_replay_coherence_delta.v1"
    )
    assert report["heldout_generation_coherence_delta"][
        "source_continuation_loss_available"
    ] is True
    assert report["fresh_heldout_prompt_suite"]["enabled"] is True
    assert report["fresh_heldout_prompt_suite"]["built_after_candidate_selection"] is True
    assert report["fresh_heldout_prompt_suite"]["case_count"] > 0
    assert report["fresh_heldout_prompt_suite"]["not_used_for_replay_training"] is True
    assert report["fresh_heldout_prompt_suite"]["training_prompt_overlap_count"] == 0
    assert report["fresh_heldout_prompt_suite"]["fixed_heldout_prompt_overlap_count"] == 0
    assert "source_text" not in report["fresh_heldout_prompt_suite"]["prompt_cases"][0]
    assert report["fresh_heldout_generation_coherence_before"]["checkpoint_path"] == (
        str(parent)
    )
    assert report["fresh_heldout_generation_coherence_after"]["checkpoint_path"] == (
        str(child)
    )
    assert report["fresh_heldout_generation_coherence_delta"][
        "source_continuation_loss_available"
    ] is True
    assert report["quality_generalization_review"]["surface"] == (
        "marulho_language_quality_replay_generalization_review.v1"
    )
    assert report["quality_generalization_review"][
        "heldout_prompt_coherence_recorded"
    ] is True
    assert report["quality_generalization_review"][
        "fresh_heldout_prompt_coherence_recorded"
    ] is True
    assert report["quality_generalization_review"][
        "fresh_heldout_case_count"
    ] == report["fresh_heldout_prompt_suite"]["case_count"]
    assert report["quality_generalization_review"][
        "same_child_controlled_decode_sustained_runtime"
    ] is True
    assert report["quality_generalization_review"][
        "same_child_controlled_decode_sustained_runtime_success"
    ] is True
    assert report["sustained_runtime_evidence_summary"]["all_success"] is True
    assert report["sustained_runtime_evidence_summary"]["report_count"] == 2
    assert report["sustained_runtime_evidence_summary"]["target_tokens"] == [2, 3]
    assert report["sustained_runtime_evidence_summary"][
        "controlled_decode_available"
    ] is True
    assert report["sustained_runtime_evidence_summary"][
        "controlled_decode_all_success"
    ] is True
    assert report["sustained_runtime_evidence_summary"][
        "controlled_decode_report_count"
    ] == 2
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
    assert report["experiment_review"][
        "records_fresh_post_selection_heldout_generation_coherence"
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
    assert suite_categories["memory_slot_runtime_impact"]["status"] == "pass"
    assert suite_categories["memory_slot_architecture_cost"]["status"] == "pass"
    assert suite_categories["memory_slot_architecture_cost"]["evidence"][
        "best_report"
    ]["candidate_slots_scored"] == 4194304
    saved_structural = suite_categories["growth_prune_safety"]["evidence"][
        "saved_structural_plasticity_evidence"
    ]
    assert saved_structural["saved_structural_plasticity_evidence_available"] is True
    assert saved_structural["proposal_kinds"] == ["memory_slot_expansion"]
    assert report["benchmark_suite_report"]["subreports"][
        "memory_slot_runtime_impact_evidence"
    ] == [str(memory_slot_runtime_impact)]
    assert report["benchmark_suite_report"]["subreports"][
        "memory_slot_architecture_cost_evidence"
    ] == [str(memory_slot_architecture_cost)]
    assert report["benchmark_suite_report"]["subreports"][
        "structural_plasticity_evidence"
    ] == [str(structural_plasticity)]
    assert report["experiment_review"]["records_hard_prompt_training_pressure"] is True
    assert report["experiment_review"]["records_candidate_child_selection"] is True
    assert report["experiment_review"]["candidate_count"] == 1
    assert report["experiment_review"]["records_same_child_generation_coherence"] is True
    assert report["experiment_review"]["records_heldout_generation_coherence"] is True
    assert report["experiment_review"]["records_same_child_sustained_runtime"] is True
    assert report["experiment_review"]["records_multiple_sustained_targets"] is True
    assert report["experiment_review"][
        "records_controlled_decode_sustained_runtime"
    ] is True
    assert report["experiment_review"][
        "same_child_controlled_decode_sustained_runtime_success"
    ] is True
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
            candidate_replay_gradient_projection_modes=("disabled", "dense_core"),
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
    assert [
        candidate["learning_config"]["replay_gradient_projection_mode"]
        for candidate in selection["candidates"]
    ] == ["disabled", "dense_core"]
    assert all(
        candidate["trained_generation_coherence_delta"][
            "source_continuation_loss_available"
        ]
        for candidate in selection["candidates"]
    )
    assert len(selected_candidates) == 1
    assert all(
        candidate["learning_update_accepted"] for candidate in selection["candidates"]
    )
    assert all(
        candidate["selection_rank_learning_acceptance"] == 1.0
        for candidate in selection["candidates"]
    )
    assert selected in {"candidate-00", "candidate-01"}
    assert Path(report["child_checkpoint_path"]).exists()
    assert Path(report["child_checkpoint_path"]).name.startswith(
        f"quality-replay-sweep-{selected}-child-checkpoint"
    )
    assert (tmp_path / "quality-replay-sweep-candidate-00-child-checkpoint.pt").exists()
    assert (tmp_path / "quality-replay-sweep-candidate-01-child-checkpoint.pt").exists()
    assert report["checkpoint_lineage"]["candidate_count"] == 2
    assert report["experiment_review"]["candidate_count"] == 2
