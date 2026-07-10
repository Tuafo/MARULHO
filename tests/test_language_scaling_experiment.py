from __future__ import annotations

import json
from pathlib import Path

from marulho.evaluation.language_scaling_experiment import (
    SURFACE,
    LanguageScalingExperimentConfig,
    ScalingArmConfig,
    _arm_token_budgets,
    _parse_arm,
    fit_language_scaling_law,
    run_language_scaling_experiment,
)
from marulho.training.language_model import load_language_model_checkpoint


def test_scaling_law_requires_a_real_grid() -> None:
    report = fit_language_scaling_law(
        [
            {
                "non_embedding_parameters": 1000.0,
                "update_tokens": 1000.0,
                "heldout_loss": 4.0,
            }
        ]
    )

    assert report["available"] is False
    assert report["model_size_count"] == 1


def test_scaling_law_fits_a_three_by_three_synthetic_grid() -> None:
    points = []
    for parameters in (1_000_000, 4_000_000, 16_000_000):
        for tokens in (1_000_000, 4_000_000, 16_000_000):
            loss = (
                2.5
                + 0.7 * (parameters / 4_000_000) ** -0.30
                + 0.5 * (tokens / 4_000_000) ** -0.25
            )
            points.append(
                {
                    "arm": str(parameters),
                    "total_parameters": float(parameters),
                    "non_embedding_parameters": float(parameters),
                    "update_tokens": float(tokens),
                    "heldout_loss": float(loss),
                }
            )

    report = fit_language_scaling_law(points)

    assert report["available"] is True
    assert report["rmse"] < 0.02
    assert report["alpha"] > 0.0
    assert report["beta"] > 0.0


def test_empirical_wall_clock_arm_scales_token_budgets() -> None:
    arm = _parse_arm("small:16:1:4:2.5")
    config = LanguageScalingExperimentConfig(
        token_budgets=(100, 200),
        budget_basis="empirical_wall_clock",
        arms=(arm, ScalingArmConfig("large", 32, 1, 4)),
    )

    assert arm.token_budget_multiplier == 2.5
    assert _arm_token_budgets(arm, config) == (250, 500)


def test_equal_token_basis_rejects_arm_multiplier() -> None:
    config = LanguageScalingExperimentConfig(
        token_budgets=(32,),
        arms=(
            ScalingArmConfig(
                "small",
                width=16,
                layers=1,
                heads=4,
                token_budget_multiplier=2.0,
            ),
        ),
    )

    try:
        run_language_scaling_experiment(
            output_path="unused.json",
            corpus_paths=("unused.txt",),
            config=config,
        )
    except ValueError as exc:
        assert "equal_update_tokens requires" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected mismatched budget basis to be rejected")


def test_single_arm_is_a_data_curve_not_a_size_decision(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    eval_corpus = tmp_path / "eval.txt"
    corpus.write_text("A small clean training stream. " * 64, encoding="utf-8")
    eval_corpus.write_text("A separate evaluation stream. " * 32, encoding="utf-8")

    report = run_language_scaling_experiment(
        output_path=tmp_path / "curve.json",
        corpus_paths=(corpus,),
        eval_corpus_path=eval_corpus,
        prompts=("An absent prompt",),
        config=LanguageScalingExperimentConfig(
            tokenizer_vocab_size=512,
            sequence_length=8,
            stride=8,
            batch_size=2,
            max_train_batches=4,
            max_eval_batches=2,
            token_budgets=(32, 64),
            arms=(ScalingArmConfig("selected", 16, 1, 4, mlp_ratio=2.0),),
            transformer_context_length=16,
            learning_rate=1.0e-3,
            precision="float32",
            generation_tokens=2,
            device="cpu",
        ),
    )

    assert report["completed_arm_count"] == 1
    assert report["selection"]["selected_arm"] == "selected"
    assert report["branch_decision"] == "continue_data_scaling_at_selected_model_size"


def test_scaling_experiment_selects_and_retains_only_best_checkpoint(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    eval_corpus = tmp_path / "eval.txt"
    corpus.write_text(
        (
            "A local language model predicts unseen continuations. "
            "Heldout loss decides which size survives.\n"
        )
        * 128,
        encoding="utf-8",
    )
    eval_corpus.write_text(
        (
            "A separate holdout contains different language. "
            "Training must never consume these windows.\n"
        )
        * 32,
        encoding="utf-8",
    )
    second_corpus = tmp_path / "second-corpus.txt"
    second_corpus.write_text(
        (
            "A second provenance shard broadens the clean training stream. "
            "Both shards share one checkpoint-owned tokenizer.\n"
        )
        * 32,
        encoding="utf-8",
    )
    output = tmp_path / "scaling.json"
    report = run_language_scaling_experiment(
        output_path=output,
        corpus_paths=(corpus, second_corpus),
        eval_corpus_path=eval_corpus,
        prompts=("This prompt is absent",),
        config=LanguageScalingExperimentConfig(
            tokenizer_vocab_size=512,
            sequence_length=8,
            stride=8,
            batch_size=2,
            max_train_batches=4,
            max_eval_batches=2,
            token_budgets=(32, 64),
            arms=(
                ScalingArmConfig("small", width=16, layers=1, heads=4, mlp_ratio=2.0),
                ScalingArmConfig("larger", width=32, layers=1, heads=4, mlp_ratio=2.0),
            ),
            transformer_context_length=16,
            learning_rate=1.0e-3,
            precision="float32",
            generation_tokens=2,
            seed=17,
            retain_best_checkpoint_only=True,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE == written["surface"]
    assert report["completed_arm_count"] == 2
    assert report["failed_arm_count"] == 0
    assert report["external_llm_used"] is False
    assert report["split"]["split_strategy"] == "explicit_text_sets"
    assert report["split"]["storage_device"] == "cpu"
    assert report["corpus"]["source_count"] == 2
    assert report["corpus"]["sources"][0]["path"] == str(corpus)
    assert report["corpus"]["sources"][1]["path"] == str(second_corpus)
    assert report["corpus"]["explicit_eval_path"] == str(eval_corpus)
    assert report["prompts"][0]["exact_prompt_absent_from_corpus"] is True
    assert report["scaling_law"]["available"] is False
    assert report["quality_boundary"]["promotes_generation_quality_claim"] is False
    retained = [
        arm for arm in report["arms"] if arm.get("checkpoint_retained") is True
    ]
    deleted = [
        arm for arm in report["arms"] if arm.get("checkpoint_retained") is False
    ]
    assert len(retained) == 1
    assert len(deleted) == 1
    assert Path(retained[0]["checkpoint_path"]).is_file()
    assert not Path(deleted[0]["checkpoint_path"]).exists()


def test_scaling_experiment_resumes_owned_training_state(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    eval_corpus = tmp_path / "eval.txt"
    corpus.write_text("Fresh training documents remain local. " * 64, encoding="utf-8")
    eval_corpus.write_text("A disjoint evaluation document. " * 32, encoding="utf-8")
    arm = ScalingArmConfig("selected", 16, 1, 4, mlp_ratio=2.0)
    base_config = LanguageScalingExperimentConfig(
        tokenizer_vocab_size=512,
        sequence_length=8,
        stride=8,
        batch_size=2,
        max_train_batches=4,
        max_eval_batches=2,
        token_budgets=(16,),
        arms=(arm,),
        transformer_context_length=16,
        learning_rate=1.0e-3,
        precision="float32",
        generation_tokens=1,
        device="cpu",
    )
    first = run_language_scaling_experiment(
        output_path=tmp_path / "first.json",
        corpus_paths=(corpus,),
        eval_corpus_path=eval_corpus,
        prompts=("Absent prompt",),
        config=base_config,
    )
    first_checkpoint = Path(first["selection"]["selected_checkpoint"])

    second = run_language_scaling_experiment(
        output_path=tmp_path / "second.json",
        corpus_paths=(corpus,),
        eval_corpus_path=eval_corpus,
        prompts=("Another absent prompt",),
        config=LanguageScalingExperimentConfig(
            **{
                **base_config.__dict__,
                "resume_checkpoint_path": str(first_checkpoint),
            }
        ),
    )

    assert second["continuation"]["enabled"] is True
    assert second["continuation"]["checkpoint_owned_by_marulho"] is True
    assert second["tokenizer"]["source"] == "checkpoint_owned"
    resumed = second["arms"][0]
    assert resumed["optimizer"]["state_restored"] is True
    assert resumed["optimizer"]["batch_order_state_restored"] is True
    assert resumed["continuation"]["prior_update_tokens"] == 16
    assert resumed["continuation"]["cumulative_update_tokens"] == 32
    second_checkpoint = Path(second["selection"]["selected_checkpoint"])
    _, _, metadata = load_language_model_checkpoint(second_checkpoint)
    assert metadata["cumulative_update_tokens"] == 32
    assert "optimizer_state" in metadata["training_state"]
