from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_exact_episodic_retrieval_screen import (
    ADVANCE_DECISION,
    ARM_NAMES,
    RelationEvalCase,
    RelationRecord,
    RetrievalScreenConfig,
    build_group_schedule,
    build_policy_rankings,
    counterfactual_behavior_metrics,
    encode_relation_records,
    gather_retrieved_sources,
    lexical_episode_rankings,
    relation_loss,
    retrieval_context_metrics,
    retrieval_screen_decision,
    selected_slots_for_mode,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _small_model(tokenizer: ByteLevelLanguageTokenizer):
    return MarulhoHashedMicroExpertLanguageModel(
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


def test_read_query_excludes_answer_before_teacher_forcing() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    bank = encode_relation_records(
        tokenizer,
        [
            RelationRecord(
                source="A fact.",
                query_prefix="Question: Q? Answer: ",
                answer="A value.",
            )
        ],
        source_length=32,
        query_length=64,
    )
    read_text = tokenizer.decode(bank.read_query_ids[0, bank.read_query_mask[0]].tolist())
    selected_targets = bank.query_target_ids[0, bank.query_loss_mask[0]]
    assert "A value" not in read_text
    assert tokenizer.decode(selected_targets.tolist()).startswith("A value.")
    assert not bool(bank.query_loss_mask[0, 0])


def test_group_schedule_and_lexical_ranking_are_label_safe() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    records = (
        RelationRecord("zircon key cup.", "Question: zircon key? Answer: ", "cup"),
        RelationRecord("wood table red.", "Question: table? Answer: ", "red"),
        RelationRecord("glass jar blue.", "Question: jar? Answer: ", "blue"),
        RelationRecord("paper box open.", "Question: box? Answer: ", "open"),
    )
    bank = encode_relation_records(
        tokenizer, records, source_length=32, query_length=48
    )
    groups = torch.tensor([[[0, 1, 2, 3]]])
    targets = torch.tensor([[0]])
    target_records = groups.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    rankings = lexical_episode_rankings(
        bank, groups, target_records, tokenizer=tokenizer
    )
    assert int(rankings[0, 0, 0]) == 0
    first, first_targets = build_group_schedule(
        record_count=16,
        steps=3,
        batch_size=2,
        facts_per_query=4,
        seed=9,
        record_labels=[f"query-{index}" for index in range(16)],
    )
    second, second_targets = build_group_schedule(
        record_count=16,
        steps=3,
        batch_size=2,
        facts_per_query=4,
        seed=9,
        record_labels=[f"query-{index}" for index in range(16)],
    )
    assert torch.equal(first, second)
    assert torch.equal(first_targets, second_targets)


def test_selected_spans_match_active_source_budget() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    records = tuple(
        RelationRecord(f"fact {index}.", f"Question: q{index}? Answer: ", str(index))
        for index in range(4)
    )
    bank = encode_relation_records(
        tokenizer, records, source_length=24, query_length=48
    )
    groups = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    lexical = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]])
    rankings = build_policy_rankings(lexical, seed=3)
    expected = {
        "off": 0,
        "all4": 4,
        "random2": 2,
        "recency2": 2,
        "lexical1": 1,
        "lexical2": 2,
    }
    for mode in ARM_NAMES:
        selected = selected_slots_for_mode(mode, rankings)
        ids, mask = gather_retrieved_sources(
            bank, groups, selected, device=torch.device("cpu")
        )
        assert ids.shape == mask.shape == (2, expected[mode] * 24)


def test_retrieved_relation_loss_reaches_full_cortex() -> None:
    torch.manual_seed(4)
    tokenizer = ByteLevelLanguageTokenizer()
    model = _small_model(tokenizer)
    retrieved = torch.randint(0, tokenizer.vocab_size, (2, 48))
    query = torch.randint(0, tokenizer.vocab_size, (2, 32))
    targets = torch.randint(0, tokenizer.vocab_size, (2, 32))
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, -4:] = True
    loss = relation_loss(model, retrieved, query, targets, mask)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_retrieval_context_metrics_keep_target_slot_metrics_only() -> None:
    cases = tuple(
        RelationEvalCase(str(index), "kind", "source", f"query-{index // 2}", ("a",), 0)
        for index in range(4)
    )
    groups = torch.tensor(
        [[0, 2, 3, 1], [1, 2, 3, 0], [2, 0, 1, 3], [3, 0, 1, 2]]
    )
    targets = torch.tensor([0, 0, 0, 0])
    lexical = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    )
    rankings = build_policy_rankings(lexical, seed=7)
    row = retrieval_context_metrics(
        "lexical1",
        group_indices=groups,
        target_slots=targets,
        policy_rankings=rankings,
        cases=cases,
        source_length=48,
    )
    assert row["selected_source_count"] == 1
    assert row["target_inclusion"] == 0.75
    assert row["target_slot_metrics_only"] is True


def _decision_rows(
    *,
    all4_candidate: float = 0.90,
    lexical2_general_delta: float = 0.05,
) -> dict[str, dict]:
    candidate = {
        "off": 0.40,
        "all4": all4_candidate,
        "random2": 0.60,
        "recency2": 0.65,
        "lexical1": 0.75,
        "lexical2": 0.87,
    }
    free = {
        "off": 0.10,
        "all4": 0.60,
        "random2": 0.20,
        "recency2": 0.25,
        "lexical1": 0.35,
        "lexical2": 0.57,
    }
    paired = {
        "off": 0.10,
        "all4": 0.60,
        "random2": 0.20,
        "recency2": 0.25,
        "lexical1": 0.35,
        "lexical2": 0.57,
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
                            lexical2_general_delta if name == "lexical2" else 0.05
                        )
                    },
                    {
                        "heldout_loss_delta": (
                            lexical2_general_delta if name == "lexical2" else 0.05
                        )
                    },
                ]
            },
        }
        for name in ARM_NAMES
    }


def test_screen_decision_requires_language_gain_and_retention() -> None:
    config = RetrievalScreenConfig()
    assert retrieval_screen_decision(
        _decision_rows(), train_steps=512, config=config
    ) == ADVANCE_DECISION
    assert retrieval_screen_decision(
        _decision_rows(), train_steps=511, config=config
    ) == "diagnostic_v21_below_screen_step_floor"
    assert retrieval_screen_decision(
        _decision_rows(all4_candidate=0.70), train_steps=512, config=config
    ) == "retire_v21_task_not_learnable_from_all_history"
    assert retrieval_screen_decision(
        _decision_rows(lexical2_general_delta=0.11),
        train_steps=512,
        config=config,
    ) == "retire_v21_lexical_retrieval_breaks_general_language"


def test_counterfactual_behavior_rejects_query_prior() -> None:
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
