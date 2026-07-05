from __future__ import annotations

import json

from marulho.evaluation.language_checkpoint_evolution_experiment import (
    ARTIFACT_KIND,
    SURFACE,
    LanguageCheckpointEvolutionExperimentConfig,
    run_language_checkpoint_evolution_experiment,
)


def test_language_checkpoint_evolution_experiment_writes_saved_evidence(
    tmp_path,
) -> None:
    output = tmp_path / (
        "checkpoint-evolution-with-long-descriptive-output-name-for-windows-path-"
        "safety.json"
    )

    report = run_language_checkpoint_evolution_experiment(
        output_path=output,
        config=LanguageCheckpointEvolutionExperimentConfig(
            embedding_dim=12,
            state_dim=20,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
            sequence_length=10,
            stride=10,
            batch_size=2,
            max_parent_eval_batches=1,
            max_child_eval_batches=1,
            max_child_train_batches=2,
            max_replay_batches=1,
            learning_rate=2e-2,
            max_steps=2,
            allow_structural_growth=True,
            route_saturation_threshold=0.0,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["promotion_gate"]["checkpoint_evolution_evidence_available"] is True
    assert report["promotion_gate"]["parent_runtime_unchanged"] is True
    assert report["promotion_gate"]["rollback_to_parent_verified"] is True
    assert report["promotion_gate"]["checkpoint_lineage_complete"] is True
    assert report["promotion_gate"][
        "long_run_evidence_required_for_parent_promotion"
    ] is True
    assert report["promotion_gate"]["promotes_parent_promotion"] is False
    assert report["checkpoint_lineage"]["lineage_complete"] is True
    assert report["checkpoint_lineage"]["child_initial_matches_parent_state"] is True
    assert report["checkpoint_lineage"]["child_final_matches_child_runtime"] is True
    assert report["evolution_review"]["isolated_child_training"] is True
    assert report["evolution_review"]["parent_kept_installed"] is True
    assert report["evolution_review"]["child_update_token_count"] > 0
    assert report["runtime_evidence"]["checkpoint_storage_device"] == "cpu"
    assert report["experiment_review"]["records_checkpoint_lineage"] is True
    assert report["experiment_review"]["records_runtime_evidence"] is True
    assert report["experiment_review"]["records_child_learning_update"] is True
    assert report["experiment_review"]["records_training_backend_policy"] is True
    assert report["split"]["used_child_train_tokens"] > 0
    assert report["split"]["used_replay_tokens"] > 0
    checkpoint_dir = output.parent / report["checkpoint_dir"]
    assert checkpoint_dir.exists()
    assert checkpoint_dir.name.startswith("evo-")
    assert len(str(checkpoint_dir.resolve())) < 160
    assert (tmp_path / "README.md").exists()
