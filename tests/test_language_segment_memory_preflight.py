from __future__ import annotations

from dataclasses import replace

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_segment_memory_preflight import (
    ADVANCE_DECISION,
    ARM_NAMES,
    RelationMemoryRecord,
    SegmentMemoryBridge,
    SegmentMemoryConfig,
    SegmentMemoryPreflightConfig,
    build_evaluation_groups,
    build_group_schedule,
    counterfactual_behavior_metrics,
    encode_relation_records,
    parse_relation_training_line,
    sample_relation_training_records,
    segment_memory_decision,
    split_relation_case_prompt,
)


def test_relation_record_parser_keeps_labels_out_of_source() -> None:
    record = parse_relation_training_line(
        "Cora put a coin in a cup. Question: Where is it? "
        "Answer: It remains in the cup."
    )
    assert record is not None
    assert record.source == "Cora put a coin in a cup."
    assert record.query_prefix == "Question: Where is it? Answer: "
    assert record.answer == "It remains in the cup."
    assert "Answer" not in record.source
    assert parse_relation_training_line("<|MARULHO_DOCUMENT|>") is None


def test_relation_case_split_removes_question_from_write_segment() -> None:
    source, query = split_relation_case_prompt(
        "Cora put a coin in a cup. Later Cora moved the cup. "
        "Where is the coin? Answer:"
    )
    assert source == "Cora put a coin in a cup. Later Cora moved the cup."
    assert query == "Question: Where is the coin? Answer: "
    assert "Where" not in source


def test_relation_sampling_is_deterministic(tmp_path) -> None:
    path = tmp_path / "relations.txt"
    path.write_text(
        "### header\n"
        "<|MARULHO_DOCUMENT|>\n"
        "A. Question: Q1? Answer: A1.\n"
        "<|MARULHO_DOCUMENT|>\n"
        "B. Question: Q2? Answer: A2.\n"
        "<|MARULHO_DOCUMENT|>\n"
        "C. Question: Q3? Answer: A3.\n",
        encoding="utf-8",
    )
    first = sample_relation_training_records(path, count=2, seed=9)
    second = sample_relation_training_records(path, count=2, seed=9)
    assert first == second
    assert len(first) == 2


def test_answer_loss_mask_starts_after_query_prefix() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    record = RelationMemoryRecord(
        source="A fact.",
        query_prefix="Question: Q? Answer: ",
        answer="A value.",
    )
    bank = encode_relation_records(
        tokenizer,
        [record],
        source_length=32,
        query_length=64,
    )
    selected = bank.query_target_ids[0, bank.query_loss_mask[0]]
    assert tokenizer.decode(selected.tolist()).startswith("A value.")
    assert not bool(bank.query_loss_mask[0, 0])


def test_segment_memory_modes_are_exact_at_zero_attachment() -> None:
    torch.manual_seed(4)
    bridge = SegmentMemoryBridge(
        SegmentMemoryConfig(width=16, slot_count=4, attention_heads=4)
    )
    query = torch.randn(2, 3, 16)
    source = torch.randn(2, 8, 5, 16)
    mask = torch.ones(2, 8, 5, dtype=torch.bool)
    for mode in ARM_NAMES:
        observed = bridge(
            mode,
            query,
            source,
            mask,
            source_segments=2,
        )
        assert torch.equal(observed, query)
    learned, learned_mask = bridge.build_memory(
        "learned", source, mask, source_segments=2
    )
    local, _local_mask = bridge.build_memory(
        "local", source, mask, source_segments=2
    )
    exact, exact_mask = bridge.build_memory(
        "exact", source, mask, source_segments=2
    )
    assert learned is not None and learned.shape == (2, 4, 16)
    assert learned_mask is not None and bool(learned_mask.all())
    assert local is not None and not torch.equal(learned, local)
    assert exact is not None and exact.shape == (2, 40, 16)
    assert exact_mask is not None and bool(exact_mask.all())
    assert bridge.bounded_state_bytes(2) == 2 * 4 * 16 * 4


def test_group_schedule_is_matched_distinct_and_deterministic() -> None:
    first = build_group_schedule(
        record_count=64,
        steps=5,
        batch_size=7,
        facts_per_example=8,
        seed=11,
    )
    second = build_group_schedule(
        record_count=64,
        steps=5,
        batch_size=7,
        facts_per_example=8,
        seed=11,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(len(set(row.tolist())) == 8 for row in first[0].reshape(-1, 8))
    labels = [f"query-{index % 8}" for index in range(64)]
    labeled = build_group_schedule(
        record_count=64,
        steps=5,
        batch_size=7,
        facts_per_example=8,
        seed=11,
        record_labels=labels,
    )[0]
    assert all(
        len({labels[index] for index in row.tolist()}) == 8
        for row in labeled.reshape(-1, 8)
    )


def test_evaluation_groups_change_only_target_within_same_query() -> None:
    labels = ["same", "same", *[f"other-{index}" for index in range(8)]]
    groups, slots = build_evaluation_groups(
        case_count=len(labels),
        facts_per_example=4,
        seed=13,
        case_labels=labels,
    )
    first = groups[0].clone()
    second = groups[1].clone()
    assert int(slots[0]) == int(slots[1])
    target_slot = int(slots[0])
    assert int(first[target_slot]) == 0
    assert int(second[target_slot]) == 1
    first[target_slot] = -1
    second[target_slot] = -1
    assert torch.equal(first, second)


def test_counterfactual_metric_requires_source_specific_output_change() -> None:
    rows = [
        {
            "query_prefix": "same",
            "expected": "red",
            "observed": "red",
            "exact": True,
        },
        {
            "query_prefix": "same",
            "expected": "blue",
            "observed": "red",
            "exact": False,
        },
    ]
    metrics = counterfactual_behavior_metrics(rows)
    assert metrics["source_following_exact_accuracy"] == 0.5
    assert metrics["output_change_rate_when_source_answer_changes"] == 0.0
    assert metrics["both_answers_correct_pair_rate"] == 0.0


def _decision_rows(
    candidate: dict[str, float],
    free: dict[str, float],
    counterfactual: dict[str, float] | None = None,
) -> dict[str, dict]:
    paired = counterfactual or free
    return {
        name: {
            "evaluation": {
                "candidate_accuracy": candidate[name],
                "free_exact_accuracy": free[name],
                "paired_counterfactual": {
                    "source_following_exact_accuracy": paired[name]
                },
            }
        }
        for name in ARM_NAMES
    }


def test_segment_memory_decision_requires_exact_and_control_margins() -> None:
    config = SegmentMemoryPreflightConfig()
    candidate = {
        "off": 0.25,
        "exact": 0.90,
        "local": 0.30,
        "recency": 0.40,
        "mean": 0.42,
        "learned": 0.75,
    }
    free = {
        "off": 0.0,
        "exact": 0.70,
        "local": 0.02,
        "recency": 0.05,
        "mean": 0.06,
        "learned": 0.35,
    }
    counterfactual = dict(free, exact=0.70, learned=0.40)
    assert segment_memory_decision(
        _decision_rows(candidate, free, counterfactual),
        train_steps=512,
        config=config,
    ) == ADVANCE_DECISION
    assert segment_memory_decision(
        _decision_rows(candidate, free, counterfactual),
        train_steps=511,
        config=config,
    ) == "diagnostic_v18b_below_preflight_step_floor"
    weak_exact = dict(candidate, exact=0.69)
    assert segment_memory_decision(
        _decision_rows(weak_exact, free, counterfactual),
        train_steps=512,
        config=config,
    ) == "retire_v18b_exact_history_no_source_causal_gain"
    matched = dict(candidate, mean=0.76)
    assert segment_memory_decision(
        _decision_rows(matched, free, counterfactual),
        train_steps=512,
        config=replace(config, minimum_learned_candidate_gain=0.0),
    ) == "retire_v18b_simple_summary_matches_learned_slots"
