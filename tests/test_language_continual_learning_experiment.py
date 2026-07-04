from __future__ import annotations

import json

from marulho.evaluation.language_continual_learning_experiment import (
    LanguageContinualLearningExperimentConfig,
    run_language_continual_learning_experiment,
)


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
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_new_batches=1,
            max_replay_batches=1,
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
