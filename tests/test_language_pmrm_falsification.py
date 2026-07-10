from __future__ import annotations

import json
from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_pmrm_falsification import (
    PMRMFalsificationConfig,
    build_matched_schedule,
    parse_pmrm_arm,
    pmrm_falsification_decision,
    run_pmrm_falsification,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def test_matched_schedule_is_deterministic_and_budgeted() -> None:
    first = build_matched_schedule(
        step_count=20,
        relation_fraction=0.20,
        relation_batch_count=3,
        general_batch_counts=(7, 5),
        seed=11,
    )
    second = build_matched_schedule(
        step_count=20,
        relation_fraction=0.20,
        relation_batch_count=3,
        general_batch_counts=(7, 5),
        seed=11,
    )
    assert first == second
    assert sum(kind == "relation" for kind, _ in first) == 4
    assert sum(kind == "general_0" for kind, _ in first) == 8
    assert sum(kind == "general_1" for kind, _ in first) == 8
    assert all(
        index < (3 if kind == "relation" else (7 if kind == "general_0" else 5))
        for kind, index in first
    )


def test_pmrm_arm_parser_keeps_ablation_inside_one_architecture() -> None:
    assert parse_pmrm_arm("transformer").architecture == "transformer"
    surprise = parse_pmrm_arm("pmrm-surprise")
    assert surprise.architecture == "pmrm"
    assert surprise.fusion_kind == "dual_parallel"
    assert surprise.episodic_policy == "surprise"
    assert parse_pmrm_arm("pmrm-none").episodic_policy == "none"
    assert parse_pmrm_arm("pmrm-temporal").fusion_kind == "temporal_only"


def test_pmrm_decision_requires_matched_quality_not_throughput() -> None:
    transformer = {
        "name": "transformer",
        "status": "completed",
        "parameters": {"total_parameters": 1000},
        "training": {"processed_tokens": 4_200_000},
        "general_holdout": {"after": {"heldout_loss": 3.0}},
        "relation": {"generation_exact_accuracy": 0.50},
    }
    pmrm = {
        "name": "pmrm-surprise",
        "status": "completed",
        "parameters": {"total_parameters": 1004},
        "training": {"processed_tokens": 4_200_000},
        "general_holdout": {"after": {"heldout_loss": 3.1}},
        "relation": {"generation_exact_accuracy": 0.65},
    }
    assert pmrm_falsification_decision((transformer, pmrm)) == (
        "scale_integrated_pmrm"
    )
    pmrm["relation"]["generation_exact_accuracy"] = 0.55
    assert pmrm_falsification_decision((transformer, pmrm)) == (
        "continue_successive_halving_or_redesign_systems_path"
    )


def test_tiny_matched_falsification_runs_both_model_adapters(tmp_path: Path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    tokenizer_checkpoint = save_language_model_checkpoint(
        tmp_path / "tokenizer-source.pt",
        MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=16,
                state_dim=16,
                state_layers=1,
                attention_heads=4,
                transformer_context_length=8,
                transformer_mlp_ratio=2.0,
            )
        ),
        tokenizer,
        metadata={"cumulative_update_tokens": 17},
    )
    relation_corpus = tmp_path / "relation.txt"
    relation_corpus.write_text("A B C D " * 100, encoding="utf-8")
    general_train = tmp_path / "general-train.txt"
    general_train.write_text("Language patterns continue. " * 100, encoding="utf-8")
    general_eval = tmp_path / "general-eval.txt"
    general_eval.write_text("Heldout patterns continue. " * 100, encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": f"{kind}-0",
                        "kind": kind,
                        "signature": f"{kind}|tiny",
                        "prompt": "A",
                        "candidates": ["B", "C"],
                        "correct_index": 0,
                    }
                    for kind in ("container", "ownership", "property", "event_order")
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pmrm.json"
    report = run_pmrm_falsification(
        tokenizer_checkpoint_path=tokenizer_checkpoint,
        relation_corpus_path=relation_corpus,
        relation_cases_path=cases,
        general_train_corpus_paths=(general_train,),
        general_eval_corpus_paths=(general_eval,),
        output_path=output,
        arms=(parse_pmrm_arm("transformer"), parse_pmrm_arm("pmrm-surprise")),
        config=PMRMFalsificationConfig(
            token_budget=32,
            relation_fraction=0.50,
            sequence_length=8,
            batch_size=2,
            eval_batches=1,
            relation_eval_batch_size=4,
            precision="float32",
            model_width=16,
            transformer_layers=1,
            attention_heads=4,
            transformer_mlp_ratio=2.0,
            column_count=4,
            active_columns=2,
            associative_dim=4,
            episodic_slots=4,
            episodic_reads=2,
            workspace_registers=2,
            workspace_layers=1,
            workspace_mlp_dim=32,
        ),
        device="cpu",
    )

    assert output.is_file()
    assert [arm["status"] for arm in report["arms"]] == ["completed", "completed"]
    assert report["tokenizer_source"]["weights_reused"] is False
    assert report["split_contract"]["identical_schedule_for_every_arm"] is True
    assert report["quality_boundary"]["promotes_runtime_installation"] is False
