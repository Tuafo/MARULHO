from marulho.evaluation.language_conditional_lora_falsification import (
    ConditionalLoRAFalsificationConfig,
    v50_gate,
)


def test_v50_gate_requires_hierarchical_gain_and_exact_inactive_path() -> None:
    config = ConditionalLoRAFalsificationConfig()
    row = {
        "processed_tokens": config.token_budget,
        "relation": {"generation_exact_accuracy": 0.89, "accuracy": 0.98},
    }
    gate = v50_gate(
        row=row,
        source={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.30},
            "intact_gain_over_stronger_control": 0.20,
        },
        parity={
            "parent_state_exact": True,
            "sample_logits_bit_exact": True,
            "heldout_loss_exact": True,
        },
        gradients={
            "all_received_gradient": True,
            "all_nonzero_gradient": True,
        },
        adapter_fraction=0.02,
        baseline_relation={"generation_exact_accuracy": 0.89, "accuracy": 0.98},
        config=config,
    )
    assert gate["passed"]


def test_v50_gate_rejects_nonzero_gradient_or_grounding_failure() -> None:
    config = ConditionalLoRAFalsificationConfig()
    row = {
        "processed_tokens": config.token_budget,
        "relation": {"generation_exact_accuracy": 0.89, "accuracy": 0.98},
    }
    gate = v50_gate(
        row=row,
        source={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.25},
            "intact_gain_over_stronger_control": 0.20,
        },
        parity={
            "parent_state_exact": True,
            "sample_logits_bit_exact": True,
            "heldout_loss_exact": True,
        },
        gradients={
            "all_received_gradient": True,
            "all_nonzero_gradient": False,
        },
        adapter_fraction=0.02,
        baseline_relation={"generation_exact_accuracy": 0.89, "accuracy": 0.98},
        config=config,
    )
    assert not gate["passed"]
    assert not gate["checks"]["minimum_gain_over_v48"]
    assert not gate["checks"]["all_adapter_parameters_nonzero_gradient"]
