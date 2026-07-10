from __future__ import annotations

from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_delta_falsification import (
    DeltaFalsificationConfig,
    _build_model,
    _load_or_build_schedule,
    build_matched_schedule,
    delta_falsification_decision,
    parse_delta_arm,
)


def test_delta_arm_parser_and_parameter_matching() -> None:
    config = DeltaFalsificationConfig(sequence_length=72)
    transformer = _build_model(
        parse_delta_arm("transformer"), vocab_size=8192, config=config
    )
    pure = _build_model(parse_delta_arm("delta"), vocab_size=8192, config=config)
    hybrid = _build_model(
        parse_delta_arm("delta-hybrid"), vocab_size=8192, config=config
    )
    hybrid_half = _build_model(
        parse_delta_arm("delta-hybrid-half"), vocab_size=8192, config=config
    )
    counts = [
        sum(parameter.numel() for parameter in model.parameters())
        for model in (transformer, pure, hybrid, hybrid_half)
    ]
    assert counts == [20_976_128, 20_978_176, 20_977_664, 20_977_152]


def test_delta_schedule_is_deterministic_and_source_balanced() -> None:
    kwargs = {
        "step_count": 10,
        "relation_fraction": 0.2,
        "relation_batch_count": 3,
        "general_batch_counts": (4, 5),
        "seed": 19,
    }
    first = build_matched_schedule(**kwargs)
    second = build_matched_schedule(**kwargs)
    assert first == second
    assert sum(kind == "relation" for kind, _ in first) == 2
    assert sum(kind == "general_0" for kind, _ in first) == 4
    assert sum(kind == "general_1" for kind, _ in first) == 4


def _decision_row(name: str, *, loss: float, tokens: int = 262_144) -> dict:
    parameters = 20_976_128 if name == "transformer" else 20_978_176
    return {
        "name": name,
        "status": "completed",
        "parameters": {"total_parameters": parameters},
        "training": {"processed_tokens": tokens},
        "general_holdout": {"after": {"heldout_loss": loss}},
    }


def test_delta_decision_selects_best_candidate_without_overclaiming() -> None:
    decision = delta_falsification_decision(
        (
            _decision_row("transformer", loss=7.50),
            _decision_row("delta", loss=7.63),
            _decision_row("delta-hybrid", loss=7.58),
        )
    )
    assert decision == "continue_delta-hybrid_to_next_budget"


def test_frozen_schedule_cache_is_content_addressed(tmp_path: Path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    relation = tmp_path / "relation.txt"
    train = tmp_path / "train.txt"
    evaluate = tmp_path / "eval.txt"
    relation.write_text("relation facts repeat enough. " * 40, encoding="utf-8")
    train.write_text("general training text repeats enough. " * 40, encoding="utf-8")
    evaluate.write_text("held out evaluation text repeats enough. " * 40, encoding="utf-8")
    cache = tmp_path / "schedule.pt"
    config = DeltaFalsificationConfig(
        token_budget=32,
        sequence_length=4,
        batch_size=2,
        eval_batches=2,
        seed=23,
    )

    first = _load_or_build_schedule(
        tokenizer=tokenizer,
        relation_corpus=relation,
        general_train=(train,),
        general_eval=(evaluate,),
        config=config,
        step_count=4,
        cache_path=cache,
    )
    second = _load_or_build_schedule(
        tokenizer=tokenizer,
        relation_corpus=relation,
        general_train=(train,),
        general_eval=(evaluate,),
        config=config,
        step_count=4,
        cache_path=cache,
    )
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["contract_hash"] == second["contract_hash"]
    assert first["schedule"] == second["schedule"]

    different_initialization = DeltaFalsificationConfig(
        token_budget=32,
        sequence_length=4,
        batch_size=2,
        eval_batches=2,
        seed=23,
        model_seed=99,
    )
    third = _load_or_build_schedule(
        tokenizer=tokenizer,
        relation_corpus=relation,
        general_train=(train,),
        general_eval=(evaluate,),
        config=different_initialization,
        step_count=4,
        cache_path=cache,
    )
    assert third["cache_hit"] is True
    assert third["contract_hash"] == first["contract_hash"]
