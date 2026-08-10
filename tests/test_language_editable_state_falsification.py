from __future__ import annotations

import torch

from marulho.evaluation.language_editable_state_falsification import (
    ARM_NAMES,
    EditableStateFalsificationConfig,
    architecture_comparison,
    build_arm_model,
    copy_shared_initialization,
    editable_state_decision,
)
from marulho.training.language_editable_state_hybrid import (
    MarulhoEditableStateHybridLanguageModel,
)


def _row(loss: float, rate: float, *, gradients: bool = True) -> dict:
    return {
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": 0.0},
        "training": {
            "tokens_per_second": rate,
            "peak_cuda_memory_bytes": 100,
        },
        "all_parameters_received_final_gradient": gradients,
        "nonzero_final_gradient_elements": 1 if gradients else 0,
        "execution": {"warmup_loss_parity": {"passed": True}},
    }


def test_v33_models_are_parameter_matched_and_share_surviving_tensors() -> None:
    config = EditableStateFalsificationConfig(
        width=32,
        layers=4,
        heads=4,
        mlp_ratio=2.0,
        sequence_length=16,
        local_attention_window=8,
        matrix_chunk_size=16,
    )
    torch.manual_seed(17)
    baseline = build_arm_model("transformer", vocab_size=96, config=config)
    torch.manual_seed(17)
    candidate = build_arm_model("editable_hybrid", vocab_size=96, config=config)
    assert isinstance(candidate, MarulhoEditableStateHybridLanguageModel)
    audit = copy_shared_initialization(baseline, candidate)
    assert audit["shared_tensors_bit_exact"] is True
    assert audit["shared_parameter_elements"] > 96 * 32
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )


def test_v33_decision_requires_durable_loss_and_viable_execution() -> None:
    config = EditableStateFalsificationConfig()
    winner = {
        "transformer": _row(4.10, 56_000.0),
        "editable_hybrid": _row(4.07, 37_000.0),
    }
    comparison = architecture_comparison(winner)
    assert comparison is not None
    assert abs(comparison["editable_hybrid_loss_gain"] - 0.03) < 1.0e-9
    assert editable_state_decision(
        winner,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        shared_initialization_passed=True,
        config=config,
    ) == "advance_v33_editable_state_to_unseen_generation"
    too_slow = {
        **winner,
        "editable_hybrid": _row(4.07, 10_000.0),
    }
    assert editable_state_decision(
        too_slow,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        shared_initialization_passed=True,
        config=config,
    ) == "retire_v33_editable_state_execution_not_viable"


def test_v33_decision_rejects_weak_or_invalid_evidence() -> None:
    config = EditableStateFalsificationConfig()
    tied = {
        "transformer": _row(4.10, 56_000.0),
        "editable_hybrid": _row(4.09, 37_000.0),
    }
    assert editable_state_decision(
        tied,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        shared_initialization_passed=True,
        config=config,
    ) == "retire_v33_editable_state_no_heldout_language_win"
    assert editable_state_decision(
        tied,
        processed_tokens=1_000_000,
        parameter_delta_fraction=0.0,
        shared_initialization_passed=True,
        config=config,
    ) == "diagnostic_v33_below_durable_token_floor"
    invalid = {**tied, "editable_hybrid": _row(4.00, 37_000.0, gradients=False)}
    assert editable_state_decision(
        invalid,
        processed_tokens=16_777_216,
        parameter_delta_fraction=0.0,
        shared_initialization_passed=True,
        config=config,
    ) == "invalid_v33_incomplete_gradient_coverage"
    assert set(ARM_NAMES) == {"transformer", "editable_hybrid"}

