from __future__ import annotations

from dataclasses import replace

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_joint_memory_token_preflight import (
    ADVANCE_DECISION,
    ARM_NAMES,
    JointMemoryTokenConfig,
    JointMemoryTokenCortex,
    JointMemoryTokenOrgan,
    RelationMemoryRecord,
    _scheduled_relation_steps,
    build_evaluation_groups,
    build_group_schedule,
    counterfactual_behavior_metrics,
    encode_relation_records,
    joint_memory_token_decision,
    parse_relation_training_line,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _small_cortex() -> tuple[JointMemoryTokenCortex, ByteLevelLanguageTokenizer]:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=tokenizer.vocab_size,
            width=32,
            layers=2,
            attention_heads=4,
            context_length=128,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=128,
            routing_heads=2,
            experts_per_head=2,
        )
    )
    memory = JointMemoryTokenOrgan(
        width=32,
        slot_count=4,
        replay_id=tokenizer.replay_id,
        initial_scale=0.03,
    )
    return (
        JointMemoryTokenCortex(
            model,
            memory,
            facts_per_example=4,
            source_segments=2,
        ),
        tokenizer,
    )


def test_relation_parser_keeps_query_and_answer_out_of_source() -> None:
    record = parse_relation_training_line(
        "Cora put a coin in a cup. Question: Where is it? "
        "Answer: It remains in the cup."
    )
    assert record is not None
    assert record.source == "Cora put a coin in a cup."
    assert record.query_prefix == "Question: Where is it? Answer: "
    assert record.answer == "It remains in the cup."
    assert "Question" not in record.source
    assert "Answer" not in record.source
    assert parse_relation_training_line("<|MARULHO_DOCUMENT|>") is None


def test_answer_loss_mask_is_ordinary_teacher_forcing_only() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    bank = encode_relation_records(
        tokenizer,
        [
            RelationMemoryRecord(
                source="A fact.",
                query_prefix="Question: Q? Answer: ",
                answer="A value.",
            )
        ],
        source_length=32,
        query_length=64,
    )
    selected = bank.query_target_ids[0, bank.query_loss_mask[0]]
    assert tokenizer.decode(selected.tolist()).startswith("A value.")
    assert not bool(bank.query_loss_mask[0, 0])
    decoded_source = tokenizer.decode(bank.source_ids[0, bank.source_mask[0]].tolist())
    assert "Question" not in decoded_source
    assert "Answer" not in decoded_source


def test_schedules_are_deterministic_distinct_and_paired() -> None:
    labels = [f"query-{index % 16}" for index in range(64)]
    first = build_group_schedule(
        record_count=64,
        steps=7,
        batch_size=5,
        facts_per_example=4,
        seed=11,
        record_labels=labels,
    )
    second = build_group_schedule(
        record_count=64,
        steps=7,
        batch_size=5,
        facts_per_example=4,
        seed=11,
        record_labels=labels,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(
        len({labels[index] for index in row.tolist()}) == 4
        for row in first[0].reshape(-1, 4)
    )
    case_labels = ["same", "same", *[f"other-{index}" for index in range(8)]]
    groups, slots = build_evaluation_groups(
        case_count=len(case_labels),
        facts_per_example=4,
        seed=13,
        case_labels=case_labels,
    )
    left = groups[0].clone()
    right = groups[1].clone()
    assert int(slots[0]) == int(slots[1])
    target_slot = int(slots[0])
    assert int(left[target_slot]) == 0
    assert int(right[target_slot]) == 1
    left[target_slot] = -1
    right[target_slot] = -1
    assert torch.equal(left, right)
    assert _scheduled_relation_steps(800, 0.75) == 600


def test_joint_cortex_modes_keep_bounded_state_and_parent_off_path() -> None:
    torch.manual_seed(4)
    cortex, tokenizer = _small_cortex()
    source = torch.randint(0, tokenizer.vocab_size, (2, 4, 12))
    query = torch.randint(0, tokenizer.vocab_size, (2, 16))
    reference = cortex.model(query, collect_telemetry=False)["logits"]
    assert torch.equal(cortex.query_logits("off", None, query), reference)
    expected_states = {
        "off": None,
        "exact": (2, 48),
        "local": None,
        "recency": (2, 4, 32),
        "mean": (2, 4, 32),
        "recurrent": (2, 4, 32),
        "partitioned": (2, 4, 32),
    }
    for mode in ARM_NAMES:
        state = cortex.build_source_state(mode, source)
        expected = expected_states[mode]
        if expected is None:
            assert state is None
        else:
            assert state is not None and tuple(state.shape) == expected
        logits = cortex.query_logits(mode, state, query)
        assert logits.shape == (2, 16, tokenizer.vocab_size)
    assert cortex.memory.state_bytes_per_stream() == 4 * 32 * 4
    assert cortex.memory.route_ids.tolist() == [
        tokenizer.replay_id + index for index in range(4)
    ]


def test_recurrent_memory_has_cross_segment_gradient_and_source_dependence() -> None:
    torch.manual_seed(7)
    cortex, tokenizer = _small_cortex()
    source = torch.randint(0, tokenizer.vocab_size, (2, 4, 12))
    changed = source.clone()
    changed[:, 0, 0] = (changed[:, 0, 0] + 1) % tokenizer.vocab_size
    first = cortex.build_source_state("recurrent", source)
    second = cortex.build_source_state("recurrent", changed)
    assert first is not None and second is not None
    assert not torch.equal(first, second)
    query = torch.randint(0, tokenizer.vocab_size, (2, 16))
    targets = torch.randint(0, tokenizer.vocab_size, (2, 16))
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, -4:] = True
    loss = cortex.relation_loss("recurrent", source, query, targets, mask)
    loss.backward()
    for name, parameter in cortex.memory.named_parameters():
        if name == "local_memory":
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, name
        assert int(torch.count_nonzero(parameter.grad)) > 0, name
    assert cortex.model.token_embedding.weight.grad is not None


def test_partitioned_memory_preserves_equal_segment_banks_and_gradients() -> None:
    torch.manual_seed(8)
    cortex, tokenizer = _small_cortex()
    source = torch.randint(0, tokenizer.vocab_size, (2, 4, 12))
    state = cortex.build_source_state("partitioned", source)
    assert state is not None and state.shape == (2, 4, 32)
    query = torch.randint(0, tokenizer.vocab_size, (2, 16))
    targets = torch.randint(0, tokenizer.vocab_size, (2, 16))
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, -4:] = True
    cortex.relation_loss("partitioned", source, query, targets, mask).backward()
    for name, parameter in cortex.memory.named_parameters():
        if name == "local_memory":
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, name
        assert int(torch.count_nonzero(parameter.grad)) > 0, name


def _decision_rows(
    *,
    exact_candidate: float = 0.90,
    recurrent_general_delta: float = 0.01,
) -> dict[str, dict]:
    candidate = {
        "off": 0.25,
        "exact": exact_candidate,
        "local": 0.30,
        "recency": 0.40,
        "mean": 0.42,
        "recurrent": 0.80,
        "partitioned": 0.88,
    }
    free = {
        "off": 0.0,
        "exact": 0.80,
        "local": 0.05,
        "recency": 0.10,
        "mean": 0.15,
        "recurrent": 0.60,
        "partitioned": 0.70,
    }
    paired = {
        "off": 0.0,
        "exact": 0.80,
        "local": 0.05,
        "recency": 0.10,
        "mean": 0.15,
        "recurrent": 0.72,
        "partitioned": 0.78,
    }
    return {
        name: {
            "evaluation": {
                "candidate_accuracy": candidate[name],
                "free_exact_accuracy": free[name],
                "paired_counterfactual": {
                    "source_following_exact_accuracy": paired[name]
                },
            },
            "general_language": {
                "sources": [
                    {
                        "heldout_loss_delta": (
                            recurrent_general_delta if name == "partitioned" else 0.01
                        )
                    },
                    {
                        "heldout_loss_delta": (
                            recurrent_general_delta if name == "partitioned" else 0.01
                        )
                    },
                ]
            },
        }
        for name in ARM_NAMES
    }


def test_decision_requires_exact_truth_recurrent_gain_and_retention() -> None:
    config = JointMemoryTokenConfig()
    rows = _decision_rows()
    assert joint_memory_token_decision(
        rows, train_steps=512, config=config
    ) == ADVANCE_DECISION
    assert joint_memory_token_decision(
        rows, train_steps=511, config=config
    ) == "diagnostic_v19_below_preflight_step_floor"
    assert joint_memory_token_decision(
        _decision_rows(exact_candidate=0.70),
        train_steps=512,
        config=config,
    ) == "retire_v19_task_not_learnable_from_exact_history"
    assert joint_memory_token_decision(
        _decision_rows(recurrent_general_delta=0.11),
        train_steps=512,
        config=config,
    ) == "retire_v19b_partitioned_memory_breaks_general_language"
    weak_partitioned = _decision_rows()
    weak_partitioned["partitioned"]["evaluation"]["paired_counterfactual"][
        "source_following_exact_accuracy"
    ] = 0.70
    assert joint_memory_token_decision(
        weak_partitioned,
        train_steps=512,
        config=replace(config, minimum_partitioned_counterfactual_accuracy=0.10),
    ) == "retire_v19b_separated_banks_do_not_beat_compressed_state"


def test_counterfactual_metric_rejects_source_independent_answer() -> None:
    metrics = counterfactual_behavior_metrics(
        [
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
    )
    assert metrics["source_following_exact_accuracy"] == 0.5
    assert metrics["output_change_rate_when_source_answer_changes"] == 0.0
    assert metrics["both_answers_correct_pair_rate"] == 0.0
