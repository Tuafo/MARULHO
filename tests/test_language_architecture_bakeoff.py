from __future__ import annotations

from marulho.evaluation.language_architecture_bakeoff import (
    REFERENCE_VARIANT,
    SURFACE,
    LanguageArchitectureBakeoffConfig,
    run_language_architecture_bakeoff,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
)


def test_language_architecture_bakeoff_compares_shared_data_and_decides(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "A wave predicts the next local signal and preserves causal order. "
            "A recurrent learner compares old evidence with unseen continuation. "
            "Sparse experts are useful only when heldout loss proves their value. "
        )
        * 40,
        encoding="utf-8",
    )
    output = tmp_path / "bakeoff.json"
    report = run_language_architecture_bakeoff(
        output_path=output,
        corpus_path=corpus,
        config=LanguageArchitectureBakeoffConfig(
            training=LanguageTrainingExperimentConfig(
                embedding_dim=8,
                state_dim=12,
                expert_count=2,
                active_expert_count=1,
                route_candidate_count=2,
                expert_hidden_dim=16,
                sequence_length=12,
                stride=6,
                batch_size=2,
                max_train_batches=2,
                max_eval_batches=2,
                learning_rate=1e-3,
                generation_tokens=2,
                sustained_target_tokens=2,
                sustained_tick_tokens=1,
                sustained_quantum_tokens=1,
                sustained_timeout_seconds=30.0,
                device="cpu",
            ),
            variants=(REFERENCE_VARIANT, "gru_routed"),
            epoch_budgets=(1,),
            seeds=(7,),
            heldout_prompt_count=2,
            heldout_prompt_characters=12,
        ),
    )

    assert report["surface"] == SURFACE
    assert report["external_llm_used"] is False
    assert report["fairness"]["same_train_split"] is True
    assert report["fairness"]["same_eval_split"] is True
    assert report["fairness"]["parameter_counts_recorded"] is True
    assert len(report["quality_curves"]) == 2
    assert report["decision"]["winner"] in {REFERENCE_VARIANT, "gru_routed"}
    assert output.exists()
    assert (tmp_path / "README.md").exists()
    assert all(
        (tmp_path / f"bakeoff-{variant}-seed7-epochs1.json").exists()
        for variant in (REFERENCE_VARIANT, "gru_routed")
    )

    reused = run_language_architecture_bakeoff(
        output_path=output,
        corpus_path=corpus,
        config=LanguageArchitectureBakeoffConfig(
            training=LanguageTrainingExperimentConfig(
                embedding_dim=8,
                state_dim=12,
                expert_count=2,
                active_expert_count=1,
                route_candidate_count=2,
                expert_hidden_dim=16,
                sequence_length=12,
                stride=6,
                batch_size=2,
                max_train_batches=2,
                max_eval_batches=2,
                learning_rate=1e-3,
                generation_tokens=2,
                sustained_target_tokens=2,
                sustained_tick_tokens=1,
                sustained_quantum_tokens=1,
                sustained_timeout_seconds=30.0,
                device="cpu",
            ),
            variants=(REFERENCE_VARIANT, "gru_routed"),
            epoch_budgets=(1,),
            seeds=(7,),
            heldout_prompt_count=2,
            heldout_prompt_characters=12,
        ),
    )
    assert reused["quality_curves"] == report["quality_curves"]


def test_language_architecture_bakeoff_accepts_transformer_falsification_arm(
    tmp_path,
) -> None:
    corpus = tmp_path / "transformer-corpus.txt"
    corpus.write_text(
        (
            "Causal attention predicts each token only from the prefix. "
            "A recurrent baseline receives the same source windows.\n"
        )
        * 40,
        encoding="utf-8",
    )
    report = run_language_architecture_bakeoff(
        output_path=tmp_path / "transformer-bakeoff.json",
        corpus_path=corpus,
        config=LanguageArchitectureBakeoffConfig(
            training=LanguageTrainingExperimentConfig(
                embedding_dim=16,
                state_dim=16,
                state_layers=1,
                attention_heads=4,
                transformer_context_length=128,
                transformer_mlp_ratio=2.0,
                expert_count=0,
                active_expert_count=1,
                sequence_length=12,
                stride=6,
                batch_size=2,
                max_train_batches=1,
                max_eval_batches=1,
                generation_tokens=1,
                sustained_target_tokens=1,
                sustained_tick_tokens=1,
                sustained_quantum_tokens=1,
                sustained_timeout_seconds=30.0,
                device="cpu",
            ),
            variants=("gru_dense", "transformer_dense"),
            epoch_budgets=(1,),
            seeds=(23,),
            heldout_prompt_count=1,
            heldout_prompt_characters=12,
        ),
    )

    assert {row["variant"] for row in report["quality_curves"]} == {
        "gru_dense",
        "transformer_dense",
    }
    assert report["decision"]["winner"] in {"gru_dense", "transformer_dense"}
    transformer = next(
        row for row in report["quality_curves"] if row["variant"] == "transformer_dense"
    )
    assert transformer["state_core"] == "transformer"
    assert transformer["external_llm_used"] is False
