from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_causal_document_retrieval_audit import (
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    encode_document_cases,
)
from marulho.evaluation.language_joint_document_retrieval_screen import (
    ADVANCE_DECISION,
    ARM_NAMES,
    JointDocumentRetrievalConfig,
    PreparedDocumentSplit,
    build_document_evaluation_schedule,
    build_document_training_schedule,
    document_task_batch,
    document_task_loss,
    joint_document_decision,
    selected_slots_for_mode,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _cases(tokenizer: ByteLevelLanguageTokenizer) -> tuple[DocumentContinuationCase, ...]:
    rows = []
    for index in range(8):
        source_index = index // 4
        base = 10 + index * 20
        rows.append(
            DocumentContinuationCase(
                case_id=str(index),
                source_index=source_index,
                source_name=f"source-{source_index}",
                document_sha256=f"hash-{index}",
                document_token_count=64,
                source_start=0,
                source_end=8,
                prefix_start=12,
                prefix_end=20,
                target_start=20,
                target_end=24,
                source_ids=tuple(
                    (base + offset) % tokenizer.vocab_size for offset in range(8)
                ),
                prefix_ids=tuple(
                    (base + 4 + offset) % tokenizer.vocab_size for offset in range(8)
                ),
                target_ids=tuple(
                    (base + 12 + offset) % tokenizer.vocab_size for offset in range(4)
                ),
            )
        )
    return tuple(rows)


def _split(tokenizer: ByteLevelLanguageTokenizer) -> PreparedDocumentSplit:
    cases = _cases(tokenizer)
    bank = encode_document_cases(
        cases,
        config=CausalDocumentRetrievalConfig(
            case_count_per_source=4,
            source_length=8,
            prefix_length=8,
            target_length=4,
            minimum_gap_tokens=4,
            maximum_gap_tokens=12,
        ),
    )
    return PreparedDocumentSplit("toy", cases, bank, ())


def _small_model(tokenizer: ByteLevelLanguageTokenizer):
    return MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=tokenizer.vocab_size,
            width=32,
            layers=2,
            attention_heads=4,
            context_length=64,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=128,
            routing_heads=2,
            experts_per_head=2,
        )
    )


def _config(**overrides) -> JointDocumentRetrievalConfig:
    values = {
        "train_cases_per_source": 4,
        "eval_cases_per_source": 4,
        "source_length": 8,
        "prefix_length": 8,
        "target_length": 4,
        "minimum_gap_tokens": 4,
        "maximum_gap_tokens": 12,
        "train_steps": 8,
        "batch_size": 2,
        "eval_batch_size": 2,
        "bootstrap_samples": 128,
        "precision": "float32",
    }
    values.update(overrides)
    return JointDocumentRetrievalConfig(**values)


def test_training_schedule_keeps_target_and_distractors_in_same_corpus() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    schedule = build_document_training_schedule(
        split,
        tokenizer,
        steps=4,
        batch_size=2,
        facts_per_query=4,
        seed=7,
    )
    assert schedule.groups.shape == (4, 2, 4)
    for step in range(4):
        for row in range(2):
            target = int(schedule.target_indices[step, row])
            group = schedule.groups[step, row].tolist()
            slot = int(schedule.target_slots[step, row])
            assert group[slot] == target
            assert len(set(group)) == 4
            assert all(
                split.cases[index].source_index == split.cases[target].source_index
                for index in group
            )
    assert set(schedule.rankings) == {
        "random2",
        "lexical1",
        "lexical2",
        "oracle2",
    }


def test_document_task_batch_matches_zero_or_one_episode_budget() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    schedule = build_document_training_schedule(
        split,
        tokenizer,
        steps=2,
        batch_size=2,
        facts_per_query=4,
        seed=3,
    )
    for mode, source_tokens in (
        ("off", 0),
        ("random2", 16),
        ("lexical1", 8),
        ("lexical2", 16),
        ("oracle2", 16),
    ):
        retrieved, query, targets, mask = document_task_batch(
            split,
            schedule,
            step_index=0,
            mode=mode,
            device=torch.device("cpu"),
        )
        assert retrieved.shape == (2, source_tokens)
        assert query.shape == targets.shape == mask.shape == (2, 11)
        assert int(mask.sum()) == 8


def test_document_task_loss_reaches_all_cortex_parameters() -> None:
    torch.manual_seed(4)
    tokenizer = ByteLevelLanguageTokenizer()
    model = _small_model(tokenizer)
    retrieved = torch.randint(0, tokenizer.vocab_size, (2, 8))
    query = torch.randint(0, tokenizer.vocab_size, (2, 11))
    targets = torch.randint(0, tokenizer.vocab_size, (2, 11))
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[:, -4:] = True
    loss = document_task_loss(model, retrieved, query, targets, mask)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_evaluation_schedule_has_label_safe_lexical_and_metrics_only_oracle() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    groups, targets, rankings = build_document_evaluation_schedule(
        split,
        tokenizer,
        facts_per_query=4,
        seed=5,
    )
    assert groups.shape == (8, 4)
    assert targets.shape == (8,)
    assert set(rankings) == {
        "random2",
        "lexical1",
        "lexical2",
        "oracle2",
        "wrong2",
    }
    oracle = selected_slots_for_mode("oracle2", rankings)
    assert oracle is not None
    assert torch.equal(oracle[:, 0], targets)
    wrong = selected_slots_for_mode("wrong2", rankings)
    assert wrong is not None
    assert not bool((wrong[:, 0] == targets).any())


def _gain(value: float) -> dict:
    return {
        "mean_loss_gain": value,
        "bootstrap_95_ci": [value - 0.001, value + 0.001],
    }


def _decision_rows(
    *,
    lexical_gain: float = 0.03,
    oracle_gain: float = 0.05,
    true_wrong_gain: float = 0.04,
    general_delta: float = 0.05,
) -> dict[str, dict]:
    losses = {
        "off": 3.0,
        "random2": 2.99,
        "lexical1": 2.98,
        "lexical2": 3.0 - lexical_gain,
        "oracle2": 3.0 - oracle_gain,
    }
    rows = {}
    for mode in ARM_NAMES:
        rows[mode] = {
            "matched_to_off": _gain(3.0 - losses[mode]),
            "general_language": {
                "sources": [
                    {"heldout_loss_delta": general_delta},
                    {"heldout_loss_delta": general_delta},
                ]
            },
            "evaluation": {
                "primary": {
                    "heldout_loss": losses[mode],
                    "target_inclusion": 0.9 if mode == "lexical2" else 1.0,
                },
                "source_use": {"true_over_wrong": _gain(true_wrong_gain)},
            },
        }
    return rows


def test_decision_requires_oracle_learnability_source_use_and_retention() -> None:
    config = _config(train_steps=512)
    assert joint_document_decision(
        _decision_rows(), train_steps=512, config=config
    ) == ADVANCE_DECISION
    assert joint_document_decision(
        _decision_rows(oracle_gain=0.005), train_steps=512, config=config
    ) == "retire_v24_document_task_not_learnable_with_oracle_history"
    assert joint_document_decision(
        _decision_rows(true_wrong_gain=0.005), train_steps=512, config=config
    ) == "retire_v24_balanced_top_two_no_joint_language_win"
    assert joint_document_decision(
        _decision_rows(general_delta=0.11), train_steps=512, config=config
    ) == "retire_v24_lexical_two_breaks_general_language"
