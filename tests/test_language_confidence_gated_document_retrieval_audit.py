from __future__ import annotations

import torch

from marulho.evaluation.language_causal_document_retrieval_audit import (
    DocumentContinuationCase,
)
from marulho.evaluation.language_confidence_gated_document_retrieval_audit import (
    ADVANCE_DECISION,
    ConfidenceGatedRetrievalConfig,
    calibrate_margin_threshold,
    compose_gated_language,
    confidence_gate_decision,
    gate_retrieval_metrics,
)


def _cases() -> tuple[DocumentContinuationCase, ...]:
    rows = []
    for index in range(8):
        source_index = index // 4
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
                source_ids=tuple(range(8)),
                prefix_ids=tuple(range(8, 16)),
                target_ids=tuple(range(16, 20)),
            )
        )
    return tuple(rows)


def _config(**overrides) -> ConfidenceGatedRetrievalConfig:
    values = {
        "calibration_cases_per_source": 4,
        "evaluation_cases_per_source": 4,
        "source_length": 8,
        "prefix_length": 8,
        "target_length": 4,
        "minimum_gap_tokens": 4,
        "maximum_gap_tokens": 12,
        "bootstrap_samples": 128,
        "precision": "float32",
    }
    values.update(overrides)
    return ConfidenceGatedRetrievalConfig(**values)


def test_calibration_freezes_maximum_coverage_high_precision_margin() -> None:
    cases = _cases()
    margins = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    rankings = torch.tensor(
        [[0, 1, 2, 3]] * 6 + [[1, 0, 2, 3], [2, 0, 1, 3]]
    )
    targets = torch.zeros(8, dtype=torch.long)
    threshold, report = calibrate_margin_threshold(
        margins,
        rankings,
        targets,
        cases,
        config=_config(),
    )
    assert threshold == torch.tensor(0.4).item()
    assert report["selected"]["selected_count"] == 6
    assert report["selected"]["precision"] == 1.0
    assert report["selection_uses_future_target_tokens"] is False
    assert report["selection_uses_language_loss"] is False


def test_gate_metrics_distinguish_coverage_precision_and_effective_recall() -> None:
    cases = _cases()
    selected = torch.tensor([[0], [0], [1], [0], [0], [1], [0], [0]])
    target = torch.zeros(8, dtype=torch.long)
    gate = torch.tensor([True, True, True, True, False, False, False, False])
    row = gate_retrieval_metrics(
        selected,
        gate,
        target,
        cases,
        source_length=8,
        oracle=False,
    )
    assert row["coverage"] == 0.5
    assert row["precision"] == 0.75
    assert row["effective_correct_retrieval_rate"] == 0.375
    assert row["mean_active_source_tokens"] == 4.0
    assert row["target_identity_metrics_only"] is True


def test_offline_gate_composition_selects_exact_per_case_paths() -> None:
    cases = _cases()
    off = {
        "case_losses": [3.0] * 8,
        "case_next_token_accuracy": [0.25] * 8,
    }
    active = {
        "case_losses": [2.0] * 8,
        "case_next_token_accuracy": [0.5] * 8,
    }
    gate = torch.tensor([True, False] * 4)
    row = compose_gated_language(
        off,
        active,
        gate,
        cases,
        config=_config(),
        seed=3,
    )
    assert row["heldout_loss"] == 2.5
    assert row["next_token_accuracy"] == 0.375
    assert row["paired_to_off"]["mean_loss_gain"] == 0.5
    assert row["offline_composition_is_exact_per_case_path_selection"] is True


def _decision_arms(
    *,
    precision: float = 0.95,
    candidate_loss: float = 2.98,
) -> dict[str, dict]:
    losses = {
        "off": 3.0,
        "lexical_always1": 2.998,
        "lexical_gated1": candidate_loss,
        "random_gated1": 2.99,
        "recency_gated1": 2.991,
        "oracle_gated1": 2.97,
    }
    rows = {}
    for name, loss in losses.items():
        gain = 3.0 - loss
        rows[name] = {
            "retrieval": {
                "coverage": 0.5 if name != "off" else 0.0,
                "precision": precision if name == "lexical_gated1" else 0.5,
            },
            "language": {
                "heldout_loss": loss,
                "paired_to_off": {
                    "mean_loss_gain": gain,
                    "bootstrap_95_ci": [gain - 0.001, gain + 0.001],
                },
                "per_source": {
                    "a": {"paired_to_off": {"mean_loss_gain": gain}},
                    "b": {"paired_to_off": {"mean_loss_gain": gain}},
                },
            },
        }
    rows["oracle_gated1"]["language"]["paired_to_off"]["bootstrap_95_ci"] = [
        0.02,
        0.04,
    ]
    return rows


def test_decision_requires_transfer_and_paired_language_win() -> None:
    calibration = {"selected": {"threshold": 0.1}}
    assert confidence_gate_decision(
        calibration=calibration,
        arms=_decision_arms(),
        config=_config(),
    ) == ADVANCE_DECISION
    assert confidence_gate_decision(
        calibration=calibration,
        arms=_decision_arms(precision=0.80),
        config=_config(),
    ) == "retire_v22b_fixed_lexical_confidence_gate_nontransfer"
    assert confidence_gate_decision(
        calibration=calibration,
        arms=_decision_arms(candidate_loss=2.997),
        config=_config(),
    ) == "retire_v22b_fixed_confidence_gate_insufficient_language_gain"
    assert confidence_gate_decision(
        calibration={"selected": None},
        arms=_decision_arms(),
        config=_config(),
    ) == "retire_v22b_no_calibration_threshold_meets_precision"
