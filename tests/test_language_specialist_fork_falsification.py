from marulho.evaluation.language_specialist_fork_falsification import (
    SpecialistForkFalsificationConfig,
    v51_gate,
)


def test_v51_gate_requires_specialist_grounding_and_exact_parent() -> None:
    config = SpecialistForkFalsificationConfig()
    gate = v51_gate(
        row={
            "processed_tokens": config.token_budget,
            "all_parameters_received_final_gradient": True,
        },
        source={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.30},
            "intact_gain_over_stronger_control": 0.20,
        },
        original_route={
            "checkpoint_file_exact": True,
            "state_exact": True,
            "general_loss_exact": True,
        },
        config=config,
    )
    assert gate["passed"]


def test_v51_gate_rejects_weak_specialist_or_parent_drift() -> None:
    config = SpecialistForkFalsificationConfig()
    gate = v51_gate(
        row={
            "processed_tokens": config.token_budget,
            "all_parameters_received_final_gradient": True,
        },
        source={
            "valid": True,
            "intact_source": {"exact_answer_accuracy": 0.25},
            "intact_gain_over_stronger_control": 0.20,
        },
        original_route={
            "checkpoint_file_exact": False,
            "state_exact": True,
            "general_loss_exact": True,
        },
        config=config,
    )
    assert not gate["passed"]
    assert not gate["checks"]["minimum_gain_over_v48"]
    assert not gate["checks"]["original_checkpoint_file_exact"]
