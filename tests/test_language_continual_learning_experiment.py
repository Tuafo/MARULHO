from __future__ import annotations

import json
import os

from marulho.evaluation.language_continual_learning_experiment import (
    LanguageContinualLearningExperimentConfig,
    _apply_training_backend_policy,
    _comparison_eval_batch_limits,
    _memory_slot_architecture_cost_comparison,
    _restore_training_backend_policy,
    _speed_sweep_candidate_output_path,
    run_language_continual_learning_experiment,
    run_language_continual_learning_speed_sweep,
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


def test_memory_slot_architecture_cost_compares_no_memory_baseline(tmp_path) -> None:
    comparison = tmp_path / "no-memory.json"
    comparison.write_text(
        json.dumps(
            {
                "model_vocab_size": 1024,
                "sampled_vocab_size": 128,
                "model_config": {
                    "memory_slot_count": 0,
                    "memory_slot_candidate_count": 0,
                    "active_memory_slot_count": 1,
                },
                "old_domain_before": {"eval_batch_count": 3},
                "new_domain_before": {"eval_batch_count": 5},
                "generation_quality_after": {
                    "mean_source_prefix_match_chars": 4.0,
                    "mean_distinct_bigram_fraction": 0.5,
                },
                "learning_evidence": {
                    "update_token_count": 4096,
                    "tokens_per_second": 4000.0,
                    "total_window_tokens_per_second": 2500.0,
                    "new_domain_loss_delta": 2.0,
                    "old_domain_forgetting": -1.0,
                    "general_replay_retention_delta": -1.5,
                    "memory_slots": {
                        "candidate_slots_scored": 0,
                        "runs_all_slots": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report = {
        "model_vocab_size": 1024,
        "sampled_vocab_size": 128,
        "model_config": {
            "memory_slot_count": 16,
            "memory_slot_candidate_count": 4,
            "active_memory_slot_count": 2,
        },
        "old_domain_before": {"eval_batch_count": 3},
        "new_domain_before": {"eval_batch_count": 5},
        "generation_quality_after": {
            "mean_source_prefix_match_chars": 6.0,
            "mean_distinct_bigram_fraction": 0.75,
        },
        "learning_evidence": {
            "update_token_count": 4096,
            "tokens_per_second": 3200.0,
            "total_window_tokens_per_second": 2400.0,
            "new_domain_loss_delta": 2.5,
            "old_domain_forgetting": -1.25,
            "general_replay_retention_delta": -1.75,
            "memory_slots": {
                "candidate_slots_scored": 32768,
                "runs_all_slots": False,
                "bounded_memory_slot_path": True,
            },
        },
    }

    cost = _memory_slot_architecture_cost_comparison(
        report,
        comparison_report_path=comparison,
    )

    assert cost["surface"] == (
        "marulho_language_continual_memory_slot_architecture_cost.v1"
    )
    assert cost["status"] == "memory_slot_architecture_cost_measured"
    assert cost["comparison_is_no_memory_baseline"] is True
    assert cost["comparable_update_throughput"] is True
    assert cost["comparable_total_window_throughput"] is True
    assert cost["delta_vs_no_memory_update_tokens_per_second"] == -800.0
    assert cost["delta_vs_no_memory_update_percent"] == -20.0
    assert cost["delta_vs_no_memory_total_window_tokens_per_second"] == -100.0
    assert cost["delta_vs_no_memory_new_domain_loss_delta"] == 0.5
    assert cost["delta_vs_no_memory_old_domain_forgetting"] == -0.25
    assert cost["delta_vs_no_memory_general_replay_retention_delta"] == -0.25
    assert cost["delta_vs_no_memory_after_mean_source_prefix_match_chars"] == 2.0
    assert cost["delta_vs_no_memory_after_mean_distinct_bigram_fraction"] == 0.25


def test_training_backend_policy_sets_and_restores_env(monkeypatch) -> None:
    monkeypatch.setenv("MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING", "1")
    monkeypatch.delenv("MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING", raising=False)

    policy = _apply_training_backend_policy(
        LanguageContinualLearningExperimentConfig(
            sampled_vocab_ce_triton_training=False,
            memory_slots_triton_training=True,
        )
    )

    assert policy["surface"] == (
        "marulho_language_continual_training_backend_policy.v1"
    )
    assert policy["previous_env"]["sampled_vocab_ce_triton_training"] == "1"
    assert policy["previous_env"]["memory_slots_triton_training"] is None
    assert os.environ["MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING"] == "0"
    assert os.environ["MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"] == "1"

    _restore_training_backend_policy(policy)

    assert os.environ["MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING"] == "1"
    assert "MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING" not in os.environ


def test_language_continual_learning_speed_sweep_writes_candidate_reports(
    tmp_path,
) -> None:
    output = tmp_path / "speed-sweep.json"
    old_corpus = tmp_path / "old.txt"
    new_corpus = tmp_path / "new.txt"
    old_corpus.write_text(
        (
            "Old replay domain keeps rollback evidence and retained language. "
            "Replay protection measures old heldout loss. "
        )
        * 16,
        encoding="utf-8",
    )
    new_corpus.write_text(
        (
            "New continual domain adapts sparse rows with replay protection. "
            "GPU evidence measures update throughput. "
        )
        * 16,
        encoding="utf-8",
    )

    report = run_language_continual_learning_speed_sweep(
        output_path=output,
        recurrent_gradient_horizons=(1, 2),
        old_corpus_path=old_corpus,
        new_corpus_path=new_corpus,
        config=LanguageContinualLearningExperimentConfig(
            model_vocab_size=512,
            sampled_vocab_size=32,
            embedding_dim=8,
            state_dim=12,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=16,
            sequence_length=8,
            stride=4,
            batch_size=2,
            max_new_batches=1,
            max_replay_batches=1,
            generation_tokens=2,
            generation_prompts=("Old replay domain",),
            max_steps=1,
            gradient_clip_interval=1,
            device="cpu",
        ),
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    candidate_one = _speed_sweep_candidate_output_path(
        output,
        candidate_index=1,
        recurrent_gradient_horizon=1,
    )
    candidate_two = _speed_sweep_candidate_output_path(
        output,
        candidate_index=2,
        recurrent_gradient_horizon=2,
    )

    assert output.exists()
    assert (tmp_path / "README.md").exists()
    assert candidate_one.exists()
    assert candidate_two.exists()
    assert report["surface"] == "marulho_language_continual_speed_sweep.v1"
    assert report["status"] == "final"
    assert report["candidate_count"] == 2
    assert report["completed_candidate_count"] == 2
    assert report["accepted_candidate_count"] >= 1
    assert loaded["best_candidate"] == report["best_candidate"]
    assert [candidate["recurrent_gradient_horizon"] for candidate in report["candidates"]] == [
        1,
        2,
    ]
    assert all(
        candidate["update_tokens_per_second"] > 0.0
        for candidate in report["candidates"]
    )
    assert all(
        candidate["tracked_triton_failures"] == 0
        for candidate in report["candidates"]
    )
    assert report["best_candidate"]["accepted_online_update"] is True
    assert report["review"]["writes_partial_after_each_candidate"] is True
    assert (
        report["review"]["candidate_reports_are_complete_continual_learning_reports"]
        is True
    )
    assert report["review"]["promotes_runtime_claim"] is False


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
            paired_sampled_vocab_loss=True,
            profile_update_stages=True,
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
    assert report["experiment_review"]["records_training_backend_policy"] is True
    assert report["experiment_review"]["records_dense_adamw_backend"] is True
    assert report["experiment_review"]["records_active_compute"] is True
    assert report["experiment_review"]["records_update_stage_profile"] is True
    assert report["experiment_review"]["records_memory_slot_path"] is True
    assert report["experiment_review"]["records_bounded_memory_slot_path"] is True
    assert report["experiment_review"]["records_memory_slot_online_update_path"] is True
    assert report["experiment_review"]["records_memory_slot_candidate_precompute"] is True
    assert report["experiment_review"]["records_memory_slot_architecture_cost"] is False
    assert report["memory_slot_architecture_cost"]["status"] == (
        "comparison_report_missing"
    )
    assert (
        report["experiment_review"]["records_memory_slot_training_window_triton_stats"]
        is True
    )
    assert (
        report["experiment_review"]["records_training_memory_slot_backend_summary"]
        is True
    )
    assert (
        report["experiment_review"]["records_training_memory_slot_triton_autograd"]
        is False
    )
    assert (
        report["experiment_review"][
            "records_training_sampled_vocab_ce_backend_summary"
        ]
        is True
    )
    assert (
        report["experiment_review"][
            "records_training_sampled_vocab_ce_triton_autograd"
        ]
        is False
    )
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
    active_compute = report["learning_evidence"]["active_compute"]
    assert active_compute["surface"] == "marulho_language_continual_active_compute.v1"
    assert report["active_compute"] == active_compute
    assert active_compute["collected_outside_measured_update_window"] is True
    assert active_compute["per_step_active_compute_dict_build"] is False
    assert active_compute["total_parameters"] > 0
    assert active_compute["active_parameters_per_token_estimate"] > 0
    assert active_compute["sampled_vocab_training"] is True
    assert active_compute["full_vocab_logits_materialized"] is False
    assert active_compute["sampled_vocab_rows_per_loss_call"] == 32
    assert active_compute["sampled_loss_active_parameters_per_loss_call_estimate"] < (
        active_compute["active_parameters_per_token_estimate"]
    )
    assert active_compute["total_columns"] == 2
    assert active_compute["active_columns_per_token"] == 1
    assert active_compute["route_candidate_rows_scored_per_token"] == 2
    assert (
        active_compute["route_candidate_rows_scored_in_measured_update_window_estimate"]
        > 0
    )
    assert active_compute["total_memory_slots"] == 4
    assert active_compute["candidate_memory_slots_per_token"] == 2
    assert active_compute["active_memory_slots_per_token"] == 1
    assert active_compute["runs_all_memory_slots"] is False
    assert active_compute["precomputed_memory_candidate_ids_used"] is True
    assert active_compute["parameter_estimate"]["surface"] == (
        "marulho_language_model_parameter_estimate.v1"
    )
    training_delta = memory_slots["training_window_memory_slot_triton_stats_delta"]
    assert training_delta["surface"] == (
        "marulho_language_memory_slots_triton_stats_delta.v1"
    )
    assert training_delta["triton_autograd_used"] is False
    assert memory_slots["training_window_memory_slot_triton_autograd_used"] is False
    assert (
        memory_slots["training_window_memory_slot_triton_autograd_forward_calls"] == 0
    )
    training_backend = report["training_memory_slot_backend_summary"]
    assert training_backend["surface"] == (
        "marulho_language_continual_training_memory_slot_backend.v1"
    )
    assert training_backend["training_window_stats_recorded"] is True
    assert training_backend["triton_autograd_used"] is False
    assert training_backend["candidate_id_source"] == (
        "precomputed_batch_memory_candidate_ids"
    )
    backend_policy = report["training_backend_policy"]
    assert backend_policy["requested"]["sampled_vocab_ce_triton_training"] is False
    assert backend_policy["requested"]["memory_slots_triton_training"] is False
    assert backend_policy["active"]["sampled_vocab_ce_triton_training"] == "0"
    assert backend_policy["active"]["memory_slots_triton_training"] == "0"
    sampled_backend = report["training_sampled_vocab_ce_backend_summary"]
    assert sampled_backend["surface"] == (
        "marulho_language_continual_training_sampled_vocab_ce_backend.v1"
    )
    assert sampled_backend["training_window_stats_recorded"] is True
    assert sampled_backend["sampled_vocab_training"] is True
    assert sampled_backend["triton_training_autograd_requested"] is False
    assert sampled_backend["triton_kernel_used"] is False
    assert report["learning_evidence"]["dense_adamw_backend"] == "default"
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
    assert report["old_domain_before"]["evidence_collection_mode"] == "last_batch_only"
    assert report["old_domain_before"]["per_batch_evidence_dict_build"] is False
    assert report["old_domain_before"]["evidence_probe_batch_tokens"] > 0
    assert report["old_domain_before"]["caller_device_transfer_calls"] == 0
    assert report["learning_evidence"]["metric_readback_mode"] == (
        "deferred_gpu_scalar_aggregation"
    )
    stage_profile = report["learning_evidence"]["update_stage_profile"]
    assert stage_profile["surface"] == (
        "marulho_language_continual_update_stage_profile.v1"
    )
    assert stage_profile["enabled"] is True
    assert stage_profile["measurement"] == "host_perf_counter"
    assert stage_profile["profile_scope"] == "measured_continual_update_window_only"
    assert stage_profile["per_stage"]["paired_forward_loss"]["count"] == (
        report["learning_evidence"]["optimizer_step_count"]
    )
    assert stage_profile["per_stage"]["backward"]["mean_ms_per_token"] >= 0.0
    assert stage_profile["per_stage"]["gradient_clip"]["mean_ms_per_token"] >= 0.0
    assert stage_profile["per_stage"]["optimizer_step"]["mean_ms_per_token"] >= 0.0
    assert stage_profile["per_stage"]["optimizer_step_total"]["count"] == (
        report["learning_evidence"]["optimizer_step_count"]
    )
    assert stage_profile["top_stage_mean_ms_per_token"]
    fusion = report["learning_evidence"]["paired_update_replay_fusion"]
    assert fusion["enabled"] is True
    assert fusion["weighted_replay_loss_preserved"] is True
    assert fusion["actual_fused_steps"] == report["learning_evidence"][
        "optimizer_step_count"
    ]
    assert fusion["separate_replay_forward_loss_calls_avoided"] == fusion[
        "actual_fused_steps"
    ]
    assert fusion["paired_sampled_vocab_loss_fused_steps"] == fusion[
        "actual_fused_steps"
    ]
    assert fusion["sampled_vocab_ce_loss_calls_avoided"] == fusion[
        "actual_fused_steps"
    ]
    paired_sampled_vocab = report["learning_evidence"]["sampled_vocab_precompute"][
        "paired_update_replay_batches"
    ]
    assert paired_sampled_vocab["enabled"] is True
    assert paired_sampled_vocab["hot_update_window_precomputed"] is True
    assert report["learning_evidence"]["measured_update_loop_model_loss_calls"] == (
        fusion["actual_fused_steps"]
    )
    assert report["experiment_review"]["records_paired_update_replay_fusion"] is True
    assert (
        report["experiment_review"]["records_paired_sampled_vocab_loss_fusion"]
        is True
    )
    assert report["experiment_review"]["records_eval_last_batch_evidence"] is True
    assert report["learning_evidence"]["hot_update_evidence_mode"] == (
        "post_window_telemetry_probe"
    )
    assert report["learning_evidence"]["per_step_evidence_dict_build"] is False
    assert (
        report["learning_evidence"]["telemetry_probe_outside_measured_window"]
        is True
    )
    assert report["learning_evidence"]["post_window_update_probe_batch_tokens"] > 0
    assert report["learning_evidence"]["post_window_replay_probe_batch_tokens"] > 0
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
