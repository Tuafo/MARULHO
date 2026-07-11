from __future__ import annotations

import torch

from marulho.data.language_tokenizer import (
    ByteLevelLanguageTokenizer,
    LANGUAGE_DOCUMENT_SEPARATOR,
)
from marulho.evaluation.language_causal_document_retrieval_audit import (
    ADVANCE_DECISION,
    ARM_NAMES,
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    build_archive_groups,
    build_document_cases,
    build_policy_rankings,
    document_retrieval_decision,
    encode_document_cases,
    evaluate_document_arm,
    gather_retrieved_episodes,
    paired_bootstrap_gain,
    retrieval_metrics_for_arm,
    selected_slots_for_arm,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _config(**overrides) -> CausalDocumentRetrievalConfig:
    values = {
        "case_count_per_source": 4,
        "facts_per_query": 4,
        "source_length": 8,
        "prefix_length": 8,
        "target_length": 4,
        "minimum_gap_tokens": 4,
        "maximum_gap_tokens": 12,
        "eval_batch_size": 2,
        "feature_batch_size": 2,
        "sample_bytes": 1024,
        "sample_range_count": 1,
        "precision": "float32",
        "bootstrap_samples": 128,
    }
    values.update(overrides)
    return CausalDocumentRetrievalConfig(**values)


def _cases(count: int = 8) -> tuple[DocumentContinuationCase, ...]:
    rows = []
    for index in range(count):
        source_index = index // (count // 2)
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
                source_ids=tuple(range(index, index + 8)),
                prefix_ids=tuple(range(index + 8, index + 16)),
                target_ids=tuple(range(index + 16, index + 20)),
            )
        )
    return tuple(rows)


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


def test_document_cut_is_strictly_causal_and_target_masked() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    text = LANGUAGE_DOCUMENT_SEPARATOR.join(
        f"document-{index}-" + chr(97 + index) * 96 for index in range(4)
    )
    config = _config()
    cases, report = build_document_cases(
        tokenizer,
        text,
        source_index=0,
        source_name="toy",
        config=config,
        seed=7,
    )
    assert len(cases) == 4
    assert report["eligible_document_count"] == 4
    assert all(case.source_end <= case.prefix_start for case in cases)
    assert all(case.prefix_end == case.target_start for case in cases)
    bank = encode_document_cases(cases, config=config)
    assert bank.query_input_ids.shape == (4, 11)
    assert torch.equal(
        bank.query_loss_mask.sum(dim=1), torch.full((4,), 4, dtype=torch.long)
    )
    assert not bool(bank.query_loss_mask[:, :7].any())


def test_archive_groups_keep_distractors_in_corpus_and_target_metrics_only() -> None:
    cases = _cases()
    groups, target_slots = build_archive_groups(cases, facts_per_query=4, seed=9)
    for index, row in enumerate(groups.tolist()):
        assert row[int(target_slots[index])] == index
        assert len(set(row)) == 4
        assert all(cases[value].source_index == cases[index].source_index for value in row)
    selected = torch.stack((target_slots, (target_slots + 1) % 4), dim=1)
    metrics = retrieval_metrics_for_arm(
        "test",
        selected_slots=selected,
        target_slots=target_slots,
        cases=cases,
        source_length=8,
    )
    assert metrics["target_inclusion"] == 1.0
    assert metrics["target_slot_metrics_only"] is True
    assert metrics["target_document_identity_used_by_selector"] is False


def test_policy_selection_preserves_active_token_budgets() -> None:
    cases = _cases()
    bank = encode_document_cases(cases, config=_config())
    groups, target_slots = build_archive_groups(cases, facts_per_query=4, seed=3)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    rankings = build_policy_rankings(
        lexical_scores=scores,
        frozen_last_scores=scores.flip(1),
        frozen_mean_scores=scores,
        target_slots=target_slots,
        seed=5,
    )
    expected = {
        "off": 0,
        "all4": 4,
        "random1": 1,
        "random2": 2,
        "recency1": 1,
        "recency2": 2,
        "lexical1": 1,
        "lexical2": 2,
        "frozen_last1": 1,
        "frozen_last2": 2,
        "frozen_mean1": 1,
        "frozen_mean2": 2,
        "oracle1": 1,
    }
    for arm in ARM_NAMES:
        selected = selected_slots_for_arm(
            arm, policy_rankings=rankings, facts_per_query=4
        )
        retrieved = gather_retrieved_episodes(
            bank, groups, selected, device=torch.device("cpu")
        )
        assert retrieved.shape == (8, expected[arm] * 8)


def test_document_loss_scores_only_hidden_future() -> None:
    torch.manual_seed(4)
    tokenizer = ByteLevelLanguageTokenizer()
    model = _small_model(tokenizer)
    cases = tuple(
        DocumentContinuationCase(
            case_id=str(index),
            source_index=0,
            source_name="toy",
            document_sha256=f"hash-{index}",
            document_token_count=64,
            source_start=0,
            source_end=8,
            prefix_start=12,
            prefix_end=20,
            target_start=20,
            target_end=24,
            source_ids=tuple(
                torch.randint(0, tokenizer.vocab_size, (8,)).tolist()
            ),
            prefix_ids=tuple(
                torch.randint(0, tokenizer.vocab_size, (8,)).tolist()
            ),
            target_ids=tuple(
                torch.randint(0, tokenizer.vocab_size, (4,)).tolist()
            ),
        )
        for index in range(4)
    )
    bank = encode_document_cases(cases, config=_config(case_count_per_source=4))
    groups = torch.tensor([[0, 1, 2, 3]] * 4)
    row = evaluate_document_arm(
        model,
        bank,
        groups,
        None,
        cases,
        batch_size=2,
        precision="float32",
    )
    assert row["case_count"] == 4
    assert row["target_token_count"] == 16
    assert len(row["case_losses"]) == 4
    assert torch.isfinite(torch.tensor(row["heldout_loss"]))


def _decision_arms(
    *,
    oracle_gain: float = 0.02,
    candidate_gain: float = 0.015,
    candidate_inclusion: float = 0.90,
) -> dict[str, dict]:
    off_loss = 3.0
    losses = {
        "off": off_loss,
        "all4": 2.99,
        "random1": 2.998,
        "random2": 2.997,
        "recency1": 2.997,
        "recency2": 2.996,
        "lexical1": 2.994,
        "lexical2": off_loss - candidate_gain,
        "frozen_last1": 3.002,
        "frozen_last2": 3.001,
        "frozen_mean1": 3.0,
        "frozen_mean2": 2.999,
        "oracle1": off_loss - oracle_gain,
    }
    rows = {}
    for arm in ARM_NAMES:
        gain = off_loss - losses[arm]
        lower = gain - 0.001
        rows[arm] = {
            "retrieval": {
                "selected_source_count": (
                    0
                    if arm == "off"
                    else (4 if arm == "all4" else (1 if arm.endswith("1") else 2))
                ),
                "target_inclusion": (
                    candidate_inclusion
                    if arm == "lexical2"
                    else (0.25 if arm.endswith("1") else 0.5)
                )
            },
            "language": {
                "heldout_loss": losses[arm],
                "paired_to_off": {
                    "mean_loss_gain": gain,
                    "bootstrap_95_ci": [lower, gain + 0.001],
                },
                "per_source": {
                    "a": {"paired_to_off": {"mean_loss_gain": gain}},
                    "b": {"paired_to_off": {"mean_loss_gain": gain}},
                },
            },
        }
    rows["oracle1"]["retrieval"]["target_inclusion"] = 1.0
    return rows


def test_decision_requires_useful_oracle_and_label_safe_control_win() -> None:
    config = _config()
    assert document_retrieval_decision(
        _decision_arms(), config=config
    ) == (ADVANCE_DECISION, "lexical2")
    assert document_retrieval_decision(
        _decision_arms(oracle_gain=0.001), config=config
    ) == ("redesign_v22_prior_episode_not_predictively_useful", None)
    assert document_retrieval_decision(
        _decision_arms(candidate_gain=0.003), config=config
    ) == ("redesign_v22_addressing_does_not_recover_useful_episode", None)
    assert document_retrieval_decision(
        _decision_arms(candidate_inclusion=0.60), config=config
    ) == ("redesign_v22_addressing_does_not_recover_useful_episode", None)


def test_paired_bootstrap_reports_positive_gain() -> None:
    row = paired_bootstrap_gain(
        [3.0, 2.0, 4.0, 3.0],
        [2.5, 1.5, 3.5, 2.5],
        samples=256,
        seed=2,
    )
    assert row["mean_loss_gain"] == 0.5
    assert row["bootstrap_95_ci"][0] == 0.5
    assert row["positive_mean_probability"] == 1.0
