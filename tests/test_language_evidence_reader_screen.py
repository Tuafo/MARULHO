from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_causal_document_retrieval_audit import (
    CausalDocumentRetrievalConfig,
    DocumentContinuationCase,
    encode_document_cases,
)
from marulho.evaluation.language_evidence_reader_screen import (
    ADVANCE_DECISION,
    ARM_NAMES,
    EvidenceReaderScreenConfig,
    PreparedDocumentSplit,
    arm_interface_and_policy,
    build_evaluation_schedule,
    build_training_schedule,
    document_batch,
    evidence_reader_decision,
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


def _config(**overrides) -> EvidenceReaderScreenConfig:
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
    return EvidenceReaderScreenConfig(**values)


def test_schedule_keeps_target_and_distractors_inside_source() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    schedule = build_training_schedule(
        split,
        tokenizer,
        steps=4,
        batch_size=2,
        facts_per_query=4,
        seed=7,
    )
    assert set(schedule.rankings) == {"random", "lexical", "oracle", "wrong"}
    for step in range(4):
        for row in range(2):
            target = int(schedule.target_indices[step, row])
            group = schedule.groups[step, row].tolist()
            slot = int(schedule.target_slots[step, row])
            assert group[slot] == target
            assert all(
                split.cases[index].source_index == split.cases[target].source_index
                for index in group
            )


def test_document_batch_maps_controls_to_zero_raw_or_separate_interfaces() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    schedule = build_training_schedule(
        split,
        tokenizer,
        steps=2,
        batch_size=2,
        facts_per_query=4,
        seed=9,
    )
    expected = {
        "gate_zero": ("gate_zero", None, 0),
        "shuffled_reader": ("separate_reader", "random", 8),
        "raw_context": ("raw_context", "lexical", 8),
        "lexical_reader": ("separate_reader", "lexical", 8),
        "oracle_reader": ("separate_reader", "oracle", 8),
    }
    for arm, (interface, policy, evidence_tokens) in expected.items():
        assert arm_interface_and_policy(arm) == (interface, policy)
        evidence, query, targets, mask, observed_interface = document_batch(
            split,
            schedule,
            step_index=0,
            arm=arm,
            device=torch.device("cpu"),
        )
        assert observed_interface == interface
        assert (0 if evidence is None else evidence.shape[1]) == evidence_tokens
        assert query.shape == targets.shape == mask.shape == (2, 11)


def test_eval_oracle_and_wrong_rankings_are_metrics_only_opposites() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = _split(tokenizer)
    groups, targets, rankings = build_evaluation_schedule(
        split,
        tokenizer,
        facts_per_query=4,
        seed=11,
    )
    oracle = rankings["oracle"][:, 0]
    wrong = rankings["wrong"][:, 0]
    assert groups.shape == (8, 4)
    assert torch.equal(oracle, targets)
    assert not bool((wrong == targets).any())


def _gain(value: float) -> dict:
    return {
        "mean_loss_gain": value,
        "bootstrap_95_ci": [value - 0.001, value + 0.001],
    }


def _decision_rows(
    *,
    candidate_loss: float = 2.96,
    general_delta: float = 0.05,
    swap_rate: float = 0.75,
) -> dict[str, dict]:
    losses = {
        "gate_zero": 3.0,
        "shuffled_reader": 2.99,
        "raw_context": 2.98,
        "lexical_reader": candidate_loss,
        "oracle_reader": 2.94,
    }
    rows = {}
    for arm in ARM_NAMES:
        loss = losses[arm]
        rows[arm] = {
            "matched_to_zero": _gain(3.0 - loss),
            "general_language": {
                "sources": [
                    {"heldout_loss_delta": general_delta},
                    {"heldout_loss_delta": general_delta},
                ]
            },
            "evaluation": {
                "primary": {
                    "heldout_loss": loss,
                    "target_inclusion": 0.75 if arm == "lexical_reader" else 1.0,
                    "per_source": {
                        "a": {"heldout_loss": loss},
                        "b": {"heldout_loss": loss},
                    },
                },
                "source_use": {
                    "true_over_wrong_reader": _gain(0.04),
                },
            },
            "generation_source_swap": {"output_change_rate": swap_rate},
        }
    return rows


def test_decision_requires_reader_to_beat_raw_and_transfer_to_generation() -> None:
    config = _config(train_steps=512)
    assert evidence_reader_decision(
        _decision_rows(), train_steps=512, config=config
    ) == ADVANCE_DECISION
    assert evidence_reader_decision(
        _decision_rows(candidate_loss=2.979), train_steps=512, config=config
    ) == "retire_v27_interleaved_reader_no_anchored_interface_gain"
    assert evidence_reader_decision(
        _decision_rows(swap_rate=0.25), train_steps=512, config=config
    ) == "retire_v27_interleaved_reader_no_anchored_interface_gain"
    assert evidence_reader_decision(
        _decision_rows(general_delta=0.11), train_steps=512, config=config
    ) == "retire_v27_interleaved_reader_breaks_general_language"
