from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_exact_episodic_retrieval_audit import (
    ADVANCE_DECISION,
    RetrievalAuditConfig,
    RetrievalCase,
    build_evaluation_groups,
    counterfactual_retrieval_metrics,
    encode_text_bank,
    frozen_feature_keys,
    lexical_tfidf_scores,
    rankings_from_scores,
    retrieval_audit_decision,
    retrieval_metrics,
    split_relation_case_prompt,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def test_case_split_keeps_question_out_of_source_write() -> None:
    source, query = split_relation_case_prompt(
        "Cora put a coin in a cup. Where is the coin? Answer:"
    )
    assert source == "Cora put a coin in a cup."
    assert query == "Question: Where is the coin? Answer: "
    assert "Where" not in source


def test_paired_groups_change_only_target_source() -> None:
    labels = ["same", "same", *[f"other-{index}" for index in range(8)]]
    groups, slots = build_evaluation_groups(
        case_count=len(labels),
        facts_per_query=4,
        seed=13,
        case_labels=labels,
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


def test_lexical_tfidf_retrieves_unique_query_entity() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    sources = encode_text_bank(
        tokenizer,
        [
            "Cora placed the zirconium key in a cup.",
            "Mira painted a wooden table.",
            "Tomas moved a glass jar.",
            "Lena opened a paper box.",
        ],
        length=64,
        add_eos=True,
    )
    queries = encode_text_bank(
        tokenizer,
        [
            "Question: Where is the zirconium key? Answer: ",
            "Question: What happened? Answer: ",
            "Question: What happened? Answer: ",
            "Question: What happened? Answer: ",
        ],
        length=64,
        add_eos=False,
    )
    groups = torch.tensor([[0, 1, 2, 3]] * 4)
    scores = lexical_tfidf_scores(
        sources,
        queries,
        groups,
        excluded_token_ids=(
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.unk_id,
            tokenizer.checkpoint_id,
            tokenizer.replay_id,
        ),
    )
    assert int(scores[0].argmax()) == 0
    assert float(scores[0, 0]) > float(scores[0, 1:].max())


def test_frozen_feature_keys_are_normalized_and_bounded() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoHashedMicroExpertLanguageModel(
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
    bank = encode_text_bank(
        tokenizer,
        ["one short source", "another source"],
        length=32,
        add_eos=True,
    )
    last, mean = frozen_feature_keys(
        model, bank, batch_size=2, precision="float32"
    )
    assert last.shape == mean.shape == (2, 32)
    assert torch.allclose(last.norm(dim=-1), torch.ones(2), atol=1.0e-5)
    assert torch.allclose(mean.norm(dim=-1), torch.ones(2), atol=1.0e-5)


def test_retrieval_metrics_require_source_specific_selection() -> None:
    cases = (
        RetrievalCase("a", "kind", "red source", "same query"),
        RetrievalCase("b", "kind", "blue source", "same query"),
        RetrievalCase("c", "kind", "other", "other query"),
        RetrievalCase("d", "kind", "other", "third query"),
    )
    groups = torch.tensor(
        [
            [0, 2, 3, 1],
            [1, 2, 3, 0],
            [2, 0, 1, 3],
            [3, 0, 1, 2],
        ]
    )
    target_slots = torch.tensor([0, 0, 0, 0])
    scores = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    rankings = rankings_from_scores(scores)
    paired = counterfactual_retrieval_metrics(
        cases=cases,
        group_indices=groups,
        target_slots=target_slots,
        rankings=rankings,
    )
    assert paired["target_following_recall_at_1"] == 0.5
    assert paired["both_targets_selected_pair_rate"] == 0.0
    row = retrieval_metrics(
        name="test",
        scores=scores,
        cases=cases,
        group_indices=groups,
        target_slots=target_slots,
        promotable=True,
    )
    assert row["recall_at_1"] == 0.75
    assert row["target_index_metrics_only"] is True


def _policy_row(recall: float, *, pair: float | None = None) -> dict:
    return {
        "recall_at_1": recall,
        "recall_at_2": min(1.0, recall + 0.15),
        "macro_query_recall_at_1": recall,
        "mean_reciprocal_rank": recall,
        "counterfactual": {
            "both_targets_selected_pair_rate": recall if pair is None else pair
        },
    }


def test_decision_selects_only_a_fixed_key_with_headroom() -> None:
    policies = {
        "random": _policy_row(0.25),
        "recency": _policy_row(0.25),
        "lexical_tfidf": _policy_row(0.90, pair=0.80),
        "frozen_last": _policy_row(0.60),
        "frozen_mean": _policy_row(0.55),
    }
    decision, selected = retrieval_audit_decision(
        policies, config=RetrievalAuditConfig()
    )
    assert decision == ADVANCE_DECISION
    assert selected == "lexical_tfidf"
    weak = dict(policies)
    weak["lexical_tfidf"] = _policy_row(0.60)
    assert retrieval_audit_decision(
        weak, config=RetrievalAuditConfig()
    ) == ("redesign_v20_no_fixed_key_retrieves_exact_episode", None)
