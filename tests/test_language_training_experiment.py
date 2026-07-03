from __future__ import annotations

import json

from marulho.evaluation.language_training_experiment import (
    SURFACE,
    LanguageTrainingExperimentConfig,
    run_language_training_experiment,
)


def test_language_training_experiment_trains_generates_and_streams(tmp_path) -> None:
    output = tmp_path / "language-experiment.json"

    report = run_language_training_experiment(
        output_path=output,
        prompts=("MARULHO", "replay evidence"),
        config=LanguageTrainingExperimentConfig(
            embedding_dim=10,
            state_dim=16,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
            sequence_length=12,
            stride=6,
            batch_size=2,
            max_train_batches=4,
            train_epochs=1,
            generation_tokens=4,
            sustained_target_tokens=3,
            sustained_tick_tokens=2,
            sustained_quantum_tokens=1,
            sustained_timeout_seconds=30.0,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["training"]["token_count"] > 0
    assert report["training"]["tokens_per_second"] > 0.0
    assert report["training"]["batch_size"] == 2
    assert report["training"]["max_tokens_per_optimizer_step"] > 12
    assert report["training"]["loss_start"] is not None
    assert report["training"]["loss_end"] is not None
    assert report["training"]["language_plif_triton"]["triton_available"] in {
        True,
        False,
    }
    assert report["training"]["plif_surrogate_triton_used"] is False
    assert report["eval_before"]["heldout_perplexity"] > 0.0
    assert report["eval_after"]["heldout_perplexity"] > 0.0
    assert len(report["generation_after"]) == 2
    assert report["generation_after"][0]["external_llm_used"] is False
    assert report["generation_after"][0]["new_token_count"] >= 1
    assert report["generation_after"][0]["quality_probe"]["printable_fraction"] >= 0.0
    assert "source_continuation" in report["generation_after"][0]["quality_probe"]
    assert report["generation_quality_after"]["generation_count"] == 2
    assert report["generation_quality_after"]["review_kind"] == (
        "source_continuation_probe_not_human_quality_review"
    )
    assert "mean_source_prefix_match_chars_delta" in report["generation_quality_delta"]
    assert report["sustained_summary"]["success"] is True
    assert report["sustained_summary"]["token_delta"] == 3
    assert report["experiment_review"]["fast_mutable_experiment"] is True
    assert report["experiment_review"]["records_actual_training"] is True
    assert report["experiment_review"]["records_actual_generation"] is True
    assert report["experiment_review"]["records_generation_quality_probe"] is True
    assert report["experiment_review"]["promotes_runtime_claim"] is False
    assert (tmp_path / "language-experiment-checkpoint.pt").exists()
    assert (tmp_path / "language-experiment-sustained.json").exists()
    assert (tmp_path / "README.md").exists()
