from __future__ import annotations

import json

from marulho.evaluation.language_continual_learning_experiment import (
    LanguageContinualLearningExperimentConfig,
    _comparison_eval_batch_limits,
    run_language_continual_learning_experiment,
)


def test_comparison_eval_batch_limits_read_comparison_report(tmp_path) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "old_domain_before": {"eval_batch_count": 3},
                "new_domain_before": {"eval_batch_count": 5},
            }
        ),
        encoding="utf-8",
    )

    disabled = _comparison_eval_batch_limits(
        comparison_report_path=comparison,
        enabled=False,
    )
    matched = _comparison_eval_batch_limits(
        comparison_report_path=comparison,
        enabled=True,
    )

    assert disabled["enabled"] is False
    assert disabled["status"] == "disabled"
    assert matched["enabled"] is True
    assert matched["status"] == "matched_comparison_eval_batch_counts"
    assert matched["old_eval_batch_limit"] == 3
    assert matched["new_eval_batch_limit"] == 5


def test_language_continual_learning_experiment_writes_deferred_eval_report(
    tmp_path,
) -> None:
    output = tmp_path / "continual-report.json"

    report = run_language_continual_learning_experiment(
        output_path=output,
        config=LanguageContinualLearningExperimentConfig(
            model_vocab_size=512,
            sampled_vocab_size=32,
            embedding_dim=12,
            state_dim=16,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
            recurrent_gradient_horizon=2,
            memory_slot_count=4,
            memory_slot_candidate_count=2,
            active_memory_slot_count=1,
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_new_batches=1,
            max_replay_batches=1,
            generation_tokens=8,
            generation_repetition_penalty=1.1,
            generation_no_repeat_ngram_size=2,
            max_steps=1,
            gradient_clip_interval=1,
            device="cpu",
        ),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert output.exists()
    assert (tmp_path / "README.md").exists()
    assert loaded["experiment_surface"] == (
        "marulho_language_continual_learning_experiment.v1"
    )
    assert report["experiment_review"]["records_eval_metric_readback"] is True
    assert report["experiment_review"]["records_sampled_vocab_training"] is True
    assert report["experiment_review"]["records_memory_slot_path"] is True
    assert report["experiment_review"]["records_bounded_memory_slot_path"] is True
    assert report["experiment_review"]["records_memory_slot_online_update_path"] is True
    assert report["experiment_review"]["records_memory_slot_candidate_precompute"] is True
    assert report["experiment_review"]["records_generation_quality_probe"] is True
    assert report["experiment_review"]["records_generation_quality_delta"] is True
    assert report["model_config"]["memory_slot_count"] == 4
    assert report["model_config"]["memory_slot_candidate_count"] == 2
    assert report["model_config"]["active_memory_slot_count"] == 1
    memory_slots = report["learning_evidence"]["memory_slots"]
    assert memory_slots["surface"] == "marulho_language_continual_memory_slots.v1"
    assert memory_slots["enabled"] is True
    assert memory_slots["total_slots"] == 4
    assert memory_slots["candidate_slot_count"] == 2
    assert memory_slots["active_slots_per_token"] == 1
    assert memory_slots["candidate_slots_scored"] > 0
    assert memory_slots["update_candidate_slots_scored"] > 0
    assert memory_slots["replay_candidate_slots_scored"] > 0
    assert memory_slots["runs_all_slots"] is False
    assert memory_slots["candidate_id_source"] == "precomputed_batch_memory_candidate_ids"
    assert memory_slots["precomputed_candidate_ids_used"] is True
    assert memory_slots["memory_gate_readback"] is False
    assert memory_slots["bounded_memory_slot_path"] is True
    precompute = report["learning_evidence"]["sampled_vocab_precompute"]
    assert precompute["new_batches"]["memory_candidate_precompute"]["enabled"] is True
    assert precompute["new_batches"]["memory_candidate_precompute"][
        "candidate_id_source"
    ] == "precomputed_batch_memory_candidate_ids"
    assert precompute["replay_batches"]["memory_candidate_precompute"]["enabled"] is True
    assert precompute["old_eval_batches"]["memory_candidate_precompute"]["enabled"] is True
    assert precompute["new_eval_batches"]["memory_candidate_precompute"]["enabled"] is True
    assert loaded["learning_evidence"]["memory_slots"]["candidate_slots_scored"] == (
        memory_slots["candidate_slots_scored"]
    )
    assert report["old_domain_before"]["metric_readback_mode"] == (
        "deferred_gpu_scalar_aggregation"
    )
    assert report["old_domain_before"]["per_batch_metric_cpu_sync"] is False
    assert report["learning_evidence"]["metric_readback_mode"] == (
        "deferred_gpu_scalar_aggregation"
    )
    assert report["learning_evidence"]["total_window_tokens_per_second"] > 0.0
    assert report["learning_evidence"]["sampled_vocab_precompute"]["new_batches"][
        "enabled"
    ] is True
    assert report["generation_quality_before"]["generation_count"] == 2
    assert report["generation_quality_after"]["generation_count"] == 2
    assert report["generation_quality_before"]["promotes_generation_quality_claim"] is False
    assert report["generation_quality_after"]["promotes_generation_quality_claim"] is False
    assert report["generation_quality_delta"]["surface"] == (
        "marulho_language_continual_generation_quality_delta.v1"
    )
    assert "next_character_match_rate_delta" in report["generation_quality_delta"]
    assert report["generation_quality_delta"]["promotes_generation_quality_claim"] is False
    assert report["generation_before"][0]["external_llm_used"] is False
    assert report["generation_after"][0]["owned_by_marulho"] is True
    decode = report["generation_after"][0]["generation_decode"]
    assert decode["repetition_penalty_applied"] is True
    assert decode["repetition_penalty"] == 1.1
    assert decode["no_repeat_ngram_applied"] is True
    assert decode["no_repeat_ngram_size"] == 2
