from marulho.evaluation.language_conditional_adapter_falsification import (
    ConditionalAdapterFalsificationConfig,
    v49_gate,
)


def _row():
    return {
        "processed_tokens": 2_096_640,
        "heldout": {"heldout_loss": 3.1},
        "relation": {
            "generation_exact_accuracy": 0.89,
            "accuracy": 0.98,
        },
    }


def test_v49_gate_requires_grounding_and_exact_inactive_path() -> None:
    config = ConditionalAdapterFalsificationConfig()
    gate = v49_gate(
        row=_row(),
        source_grounding={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.30},
            "intact_gain_over_stronger_control": 0.20,
        },
        inactive_parity={
            "parent_state_exact": True,
            "sample_logits_bit_exact": True,
        },
        adapter_gradients={"all_received_gradient": True},
        adapter_state={"bounded": True},
        adapter_parameter_fraction=0.02,
        baseline_relation={
            "generation_exact_accuracy": 0.89,
            "accuracy": 0.98,
        },
        baseline_heldout_loss=3.1,
        config=config,
    )

    assert gate["passed"]


def test_v49_gate_rejects_v48_tie_or_parent_drift() -> None:
    config = ConditionalAdapterFalsificationConfig()
    gate = v49_gate(
        row=_row(),
        source_grounding={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.25},
            "intact_gain_over_stronger_control": 0.20,
        },
        inactive_parity={
            "parent_state_exact": False,
            "sample_logits_bit_exact": True,
        },
        adapter_gradients={"all_received_gradient": True},
        adapter_state={"bounded": True},
        adapter_parameter_fraction=0.02,
        baseline_relation={
            "generation_exact_accuracy": 0.89,
            "accuracy": 0.98,
        },
        baseline_heldout_loss=3.1,
        config=config,
    )

    assert not gate["passed"]
    assert not gate["checks"]["minimum_gain_over_v48"]
    assert not gate["checks"]["parent_state_exact"]
