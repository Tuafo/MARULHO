from __future__ import annotations

import json
from pathlib import Path

from marulho.evaluation.language_scaling_experiment import (
    SURFACE,
    LanguageScalingExperimentConfig,
    ScalingArmConfig,
    fit_language_scaling_law,
    run_language_scaling_experiment,
)


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


def test_scaling_experiment_selects_and_retains_only_best_checkpoint(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "A local language model predicts unseen continuations. "
            "Heldout loss decides which size survives.\n"
        )
        * 128,
        encoding="utf-8",
    )
    output = tmp_path / "scaling.json"
    report = run_language_scaling_experiment(
        output_path=output,
        corpus_path=corpus,
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
