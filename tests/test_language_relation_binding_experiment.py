from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_relation_binding_experiment import (
    KINDS,
    _heldout_signature,
    evaluate_relation_binding_cases,
    evaluate_relation_binding_cases_batched,
    materialize_relation_binding_benchmark,
    relation_binding_branch_decision,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def test_relation_benchmark_keeps_training_and_evaluation_signatures_disjoint(
    tmp_path: Path,
) -> None:
    corpus, cases_path, cases = materialize_relation_binding_benchmark(
        corpus_path=tmp_path / "relations.txt",
        cases_path=tmp_path / "relations.json",
        train_document_count=80,
        eval_cases_per_kind=3,
        seed=11,
    )

    assert corpus.is_file()
    assert cases_path.is_file()
    assert len(cases) == 3 * len(KINDS)
    assert all(_heldout_signature(case.signature) for case in cases)
    assert {case.kind for case in cases} == set(KINDS)
    text = corpus.read_text(encoding="utf-8")
    assert all(case.prompt not in text for case in cases)


def test_relation_prediction_scores_candidates_without_reading_label(
    tmp_path: Path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=512,
            transformer_mlp_ratio=2.0,
        )
    )
    _, _, cases = materialize_relation_binding_benchmark(
        corpus_path=tmp_path / "relations.txt",
        cases_path=tmp_path / "relations.json",
        train_document_count=4,
        eval_cases_per_kind=1,
        seed=19,
    )
    report = evaluate_relation_binding_cases(model, tokenizer, cases)

    assert report["case_count"] == len(KINDS)
    assert report["prediction_uses_correct_index"] is False
    assert report["correct_index_metrics_only"] is True
    assert all(row["label_used_for_prediction"] is False for row in report["rows"])
    assert all(row["label_used_for_generation"] is False for row in report["rows"])
    assert 0.0 <= report["generation_exact_accuracy"] <= 1.0


def test_batched_relation_evaluation_matches_serial_predictions(
    tmp_path: Path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=512,
            transformer_mlp_ratio=2.0,
        )
    ).eval()
    _, _, cases = materialize_relation_binding_benchmark(
        corpus_path=tmp_path / "relations.txt",
        cases_path=tmp_path / "relations.json",
        train_document_count=8,
        eval_cases_per_kind=2,
        seed=29,
    )
    serial = evaluate_relation_binding_cases(model, tokenizer, cases)
    batched = evaluate_relation_binding_cases_batched(
        model, tokenizer, cases, batch_size=8
    )

    assert batched["evaluation_mode"] == "length_grouped_batched"
    assert [row["predicted_index"] for row in batched["rows"]] == [
        row["predicted_index"] for row in serial["rows"]
    ]
    assert [row["generation_continuation"] for row in batched["rows"]] == [
        row["generation_continuation"] for row in serial["rows"]
    ]


def test_relation_branch_prioritizes_catastrophic_forgetting() -> None:
    assert relation_binding_branch_decision(
        accuracy_before=0.45,
        accuracy_after=0.88,
        general_loss_delta=5.4,
    ) == "relation_learned_but_catastrophic_forgetting_test_replay"


def test_relation_replay_requires_both_accuracy_and_retention() -> None:
    assert relation_binding_branch_decision(
        accuracy_before=0.48,
        accuracy_after=0.82,
        general_loss_delta=0.08,
        replay_enabled=True,
        generation_accuracy_after=0.80,
    ) == "replay_preserves_language_and_relations"
    assert relation_binding_branch_decision(
        accuracy_before=0.48,
        accuracy_after=0.84,
        general_loss_delta=0.70,
        replay_enabled=True,
        generation_accuracy_after=0.80,
    ) == "replay_insufficient_test_parameter_isolation_or_episodic_memory"
    assert relation_binding_branch_decision(
        accuracy_before=0.48,
        accuracy_after=0.98,
        general_loss_delta=0.0,
        replay_enabled=True,
        generation_accuracy_after=0.10,
    ) == "replay_improves_candidate_ranking_not_free_binding"
