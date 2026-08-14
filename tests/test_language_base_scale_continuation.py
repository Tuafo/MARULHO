from __future__ import annotations

import pytest

from marulho.evaluation.language_base_scale_continuation import (
    PHASE_TOKENS,
    REFERENCE_V77_LATER_LOSS,
    REFERENCE_V77_SOURCE_LOSSES,
    TARGET_CUMULATIVE_TOKENS,
    TRAIN_STEPS,
    WARMUP_STEPS,
    _learning_rate,
    aggregate_v78_unseen_generation,
    qualification_checks,
)


def test_v78_budget_and_learning_rate_are_frozen() -> None:
    assert PHASE_TOKENS == 31_457_280
    assert TARGET_CUMULATIVE_TOKENS == 257_429_760
    assert _learning_rate(0) == pytest.approx(3.0e-5 + 2.7e-4 / WARMUP_STEPS)
    assert _learning_rate(WARMUP_STEPS - 1) == pytest.approx(3.0e-4)
    assert _learning_rate(TRAIN_STEPS - 1) == pytest.approx(3.0e-5)


def test_v78_quality_gate_requires_total_and_per_source_gains(monkeypatch) -> None:
    import marulho.evaluation.language_base_scale_continuation as module

    monkeypatch.setattr(module, "EXPECTED_CONTRACT_SHA256", "contract")
    monkeypatch.setattr(module, "EXPECTED_SCHEDULE_SHA256", "schedule")
    monkeypatch.setattr(module, "SELECTED_PHYSICAL_BATCH", 8)
    initial = {
        "later_segment_loss": REFERENCE_V77_LATER_LOSS,
        "later_loss_by_source": REFERENCE_V77_SOURCE_LOSSES,
    }
    final = {
        "later_segment_loss": REFERENCE_V77_LATER_LOSS - 0.081,
        "later_loss_by_source": {
            name: value - 0.031
            for name, value in REFERENCE_V77_SOURCE_LOSSES.items()
        },
    }
    training = {
        "schedule_sha256": "schedule",
        "gradient_audit": {"passed": True},
        "model_state_finite": True,
        "positions": PHASE_TOKENS,
        "physical_batch": 8,
        "peak_cuda_allocated_bytes": 4 * 1024**3,
    }
    data = {
        "contract_sha256": "contract",
        "tokenizer_sha256": module.TOKENIZER_SHA256,
    }
    assert all(
        qualification_checks(
            initial=initial,
            final=final,
            training=training,
            data=data,
        ).values()
    )
    final["later_loss_by_source"]["cosmopedia_v2"] += 0.01
    checks = qualification_checks(
        initial=initial,
        final=final,
        training=training,
        data=data,
    )
    assert checks["final_improves_by_0_08"]
    assert not checks["cosmopedia_improves_by_0_03"]


def test_v78_unseen_aggregation_is_reusable_without_changing_v77_contract() -> None:
    assert callable(aggregate_v78_unseen_generation)
