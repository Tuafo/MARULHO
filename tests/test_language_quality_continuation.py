from __future__ import annotations

import pytest

from marulho.evaluation.language_quality_continuation import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    PHASE_TOKENS,
    REFERENCE_IMMUTABLE_LATER_LOSS,
    REFERENCE_STATIC_LATER_LOSS,
    REFERENCE_STATIC_SOURCE_LOSSES,
    TOKENIZER_SHA256,
    TRAIN_STEPS,
    _learning_rate,
    qualification_checks,
    unseen_generation_decision,
)


def _evaluation(later: float, fineweb: float, cosmopedia: float) -> dict:
    return {
        "later_segment_loss": later,
        "later_loss_by_source": {
            "fineweb_edu": fineweb,
            "cosmopedia_v2": cosmopedia,
        },
    }


def _training() -> dict:
    return {
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "positions": PHASE_TOKENS,
        "peak_cuda_allocated_bytes": 4 * 1024**3,
        "gradient_audit": {"passed": True},
    }


def test_v77_learning_rate_reproduces_frozen_schedule() -> None:
    assert _learning_rate(0) == pytest.approx(3.0e-5 + 2.7e-4 / 13.0)
    assert _learning_rate(12) == pytest.approx(3.0e-4)
    assert _learning_rate(TRAIN_STEPS - 1) == pytest.approx(3.0e-5)
    with pytest.raises(ValueError):
        _learning_rate(TRAIN_STEPS)


def test_v77_qualification_requires_the_full_contract() -> None:
    checks = qualification_checks(
        initial=_evaluation(REFERENCE_IMMUTABLE_LATER_LOSS, 4.19, 3.73),
        candidate=_evaluation(
            REFERENCE_STATIC_LATER_LOSS,
            REFERENCE_STATIC_SOURCE_LOSSES["fineweb_edu"],
            REFERENCE_STATIC_SOURCE_LOSSES["cosmopedia_v2"],
        ),
        training=_training(),
        parameter_count=100_679_424,
        contract_sha256=EXPECTED_CONTRACT_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
    )
    assert all(checks.values())


def test_v77_qualification_rejects_loss_only_result_with_bad_fidelity_inputs() -> None:
    training = _training()
    training["gradient_audit"] = {"passed": False}
    checks = qualification_checks(
        initial=_evaluation(REFERENCE_IMMUTABLE_LATER_LOSS, 4.19, 3.73),
        candidate=_evaluation(
            REFERENCE_STATIC_LATER_LOSS,
            REFERENCE_STATIC_SOURCE_LOSSES["fineweb_edu"],
            REFERENCE_STATIC_SOURCE_LOSSES["cosmopedia_v2"],
        ),
        training=training,
        parameter_count=100_679_423,
        contract_sha256="wrong",
        tokenizer_sha256="wrong",
    )
    assert checks["candidate_later_loss_reproduced"]
    assert not checks["parameter_count_exact"]
    assert not checks["data_contract_exact"]
    assert not checks["tokenizer_exact"]
    assert not checks["all_gradients_complete"]


def test_v77_unseen_decision_keeps_continual_learning_behind_coherence() -> None:
    assert unseen_generation_decision(
        validity_passed=True, human_review_coherent=False
    ) == "continue_base_language_training_before_continual_learning"
    assert unseen_generation_decision(
        validity_passed=True, human_review_coherent=True
    ) == "advance_v77_to_continual_learning_validation"
    assert unseen_generation_decision(
        validity_passed=False, human_review_coherent=True
    ) == "reject_v77_unseen_generation_invalid_evidence"


def test_v77_document_selection_rejects_negative_skip() -> None:
    from pathlib import Path

    from marulho.evaluation.language_quality_continuation import _select_documents

    with pytest.raises(ValueError, match="nonnegative"):
        _select_documents(Path("unused"), count=1, tokenizer=object(), skip_eligible=-1)
